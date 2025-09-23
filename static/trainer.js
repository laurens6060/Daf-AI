// ========================================================================
// Trainer UI-script voor dataset-opbouw, detectie en training (YOLOv8)
// - Beheert state (annotatie-items, polling, jobId)
// - Bouwt interactieve kaarten waar je positive/negative boxes tekent
// - Exporteert naar YOLO-formaat en start /api/train job
// - Pollt jobstatus, toont logs en downloadlink
// - Kan server-side detecties uitvoeren op de getekende items
// - Laadt basismodellen, exportdir en "resume" checkpoints
// - Ontvangt live uploads via WebSocket (type: trainer_image)
// ========================================================================

// ============ State ============
// Centrale in-memory status die we in alle handlers gebruiken.
const state = {
    items: [],   // [{ name, imgEl, boxes:[{x,y,w,h,label}], canvas, ctx, w, h }, ...]
    polling: null, // setInterval-id (als polling actief is)
    jobId: null    // huidige training job id (string) of null
};

// Snelkoppelingen naar DOM-elementen (eenmalig zoeken, later hergebruiken)
const els = {
    // Sidebar / knoppen / infovelden
    thumbs:    document.getElementById('thumbs'),     // container met alle image-kaarten
    kPos:      document.getElementById('kPos'),       // KPI: aantal positieve boxes
    kNeg:      document.getElementById('kNeg'),       // KPI: aantal negatieve boxes
    kImgs:     document.getElementById('kImgs'),      // KPI: aantal afbeeldingen
    log:       document.getElementById('log'),        // textarea/pre met loglijnen
    status:    document.getElementById('status'),     // status-badge (Bootstrap)
    jobInfo:   document.getElementById('jobInfo'),    // klein infoveld met job-id

    // Bestandskeuze + knoppen
    file:      document.getElementById('file'),       // <input type="file" multiple>
    pick:      document.getElementById('pick'),       // "Kies bestanden" (klik triggert file.click())
    sample:    document.getElementById('sample'),     // knop om voorbeeldimages te laden
    clear:     document.getElementById('clear'),      // knop om alle kaarten te verwijderen

    // Train/Export knoppen
    exportTrain: document.getElementById('exportTrain'), // start export & training
    cancelPoll:  document.getElementById('cancelPoll'),  // stop polling handmatig

    // Formvelden voor training
    className: document.getElementById('className'),  // classnaam (YOLO names: [className])
    modelSel:  document.getElementById('modelSel'),   // basismodel key (yolov8n/s/…)
    exportDir: document.getElementById('exportDir'),  // exportpad (optioneel)

    // Resume vanaf bestaand model
    resumeSel: document.getElementById('resumeSel'),  // <select> met bestaande .pt's
    loadResume: document.getElementById('loadResumeBtn') // knop "Load model" zonder training
};

// Kleine helpers voor logging en statusweergave
function log(msg) {
    // Voeg een regel toe aan het logpaneel en scroll naar beneden
    els.log.textContent += `\n${msg}`;
    els.log.scrollTop = els.log.scrollHeight;
}
function setStatus(text, badge = 'text-bg-secondary') {
    // Zet de Bootstrap-badge op de gewenste kleur + tekst
    els.status.className = 'badge ' + badge;
    els.status.textContent = text;
}
function refreshKpis() {
    // Tel cumulatieve positive/negative boxes + images en update KPI’s
    let p = 0, n = 0, imgs = state.items.length;
    state.items.forEach(it => it.boxes.forEach(b => b.label === 'neg' ? n++ : p++));
    els.kPos.textContent = p;
    els.kNeg.textContent = n;
    els.kImgs.textContent = imgs;
}

// ============ Geometry ============
// Eenvoudige IoU (Intersection over Union) voor rechthoeken in (x,y,w,h) vorm.
// Handig indien je overlappende boxes wilt dedupliceren of quality checks wilt doen.
function iou(a, b) {
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = a.w * a.h, areaB = b.w * b.h;
    return inter / (areaA + areaB - inter + 1e-6);
}

// ============ UI: kaart maken ============
// Voegt een image-kaart toe met een IMG en een overlay CANVAS waar je boxes kunt tekenen.
function addImageCard(fileName, imgUrl) {
    // Kolom + card-shell
    const col = document.createElement('div'); col.className = 'col-12 col-md-6';
    const card = document.createElement('div'); card.className = 'image-card shadow-sm';

    // Header met bestandsnaam (afgekapt wanneer te lang)
    const header = document.createElement('div'); header.className = 'd-flex justify-content-between align-items-center p-2 border-bottom bg-light';
    header.innerHTML = `<strong class="text-truncate">${fileName}</strong>`;

    // Wrap voor image + canvas (canvas staat bovenop de afbeelding)
    const canvasWrap = document.createElement('div'); canvasWrap.className = 'canvas-wrap';
    const img = new Image(); img.src = imgUrl; img.crossOrigin = 'anonymous'; // crossOrigin voor lokale files/WS uploads
    const canvas = document.createElement('canvas'); const ctx = canvas.getContext('2d');

    canvasWrap.appendChild(img);     // onderlaag
    canvasWrap.appendChild(canvas);  // bovenlaag (tekenlaag)
    card.appendChild(header);
    card.appendChild(canvasWrap);
    col.appendChild(card);
    els.thumbs.prepend(col);         // nieuwste bovenaan

    // Item in state registreren
    const item = { name: fileName, imgEl: img, boxes: [], canvas, ctx, w: 0, h: 0 };
    state.items.push(item);

    // Wanneer de afbeelding geladen is, canvas op ware resolutie zetten en tekenen
    img.onload = () => {
        item.w = img.naturalWidth;
        item.h = img.naturalHeight;
        canvas.width = item.w;
        canvas.height = item.h;
        drawItem(item);
        refreshKpis();
    };

    // Interactieve annotatie: muis down -> start, move -> huidige rechthoek, mouseup -> box vastleggen
    let drawing = false, start = null, cur = null;

    // Schermcoördinaten (client) omzetten naar "natuurlijke" canvas-coördinaten (ware resolutie)
    function toNatural(e) {
        const r = canvas.getBoundingClientRect();
        const cx = (e.clientX - r.left) * canvas.width / r.width;
        const cy = (e.clientY - r.top) * canvas.height / r.height;
        return { x: cx, y: cy };
    }

    // Start tekenen
    canvas.addEventListener('mousedown', e => {
        drawing = true;
        start = toNatural(e);
    });

    // Tijdens tekenen: laat een gele "ghost" box zien
    canvas.addEventListener('mousemove', e => {
        if (!drawing) return;
        cur = toNatural(e);
        drawItem(item, start, cur);
    });

    // Mouseup overal (ook buiten canvas) laat box vallen
    window.addEventListener('mouseup', e => {
        if (!drawing) return;
        drawing = false;
        const end = toNatural(e);

        // Normaliseer naar linkerboven + positieve w/h
        const x = Math.min(start.x, end.x), y = Math.min(start.y, end.y);
        const w = Math.abs(end.x - start.x), h = Math.abs(end.y - start.y);

        // Minimale boxgrootte om misclicks te vermijden
        if (w > 10 && h > 10) {
            // SHIFT ingedrukt = negatieve box (b.v. hard negatives), anders positief
            const isNeg = e.shiftKey;
            item.boxes.push({ x, y, w, h, label: isNeg ? 'neg' : 'pos' });
            drawItem(item);
            refreshKpis();
            log(`${isNeg ? '- Negatieve' : '+ Positieve'} box toegevoegd (${fileName})`);
        }
    });

    // Rechtermuisklik = undo (laatste box verwijderen)
    canvas.addEventListener('contextmenu', e => {
        e.preventDefault();
        if (item.boxes.length) {
            item.boxes.pop();
            drawItem(item);
            refreshKpis();
            log(`− Laatste box verwijderd (${fileName})`);
        }
    });

    return item;
}

// ============ Tekenen ============
// Teken image + alle definitieve boxes. Optioneel ook een "ghost" selectie tijdens het slepen.
function drawItem(item, start = null, cur = null) {
    const { ctx, canvas, imgEl } = item;

    // Achtergrond resetten en bronafbeelding tekenen
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgEl, 0, 0, item.w, item.h);

    // Lijndikte groeit mee met resolutie, zodat het er consistent uitziet
    ctx.lineWidth = Math.max(4, Math.round(canvas.width / 350));
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    // Definitieve boxes tekenen (groen = positief, rood = negatief)
    item.boxes.forEach(b => {
        const x = b.x, y = b.y, w = b.w, h = b.h;

        // Transparante vulling + slagschaduw
        ctx.fillStyle = (b.label === 'neg') ? 'rgba(239,68,68,0.15)' : 'rgba(0,255,195,0.18)';
        ctx.fillRect(x, y, w, h);
        ctx.shadowColor = 'rgba(0,0,0,0.7)';
        ctx.shadowBlur = 6;

        // Contour
        ctx.strokeStyle = (b.label === 'neg') ? '#ef4444' : '#00ffc3';
        ctx.strokeRect(x, y, w, h);
        ctx.shadowBlur = 0;
    });

    // "Ghost" selectie (geel) terwijl de muis beweegt
    if (start && cur) {
        const x = Math.min(start.x, cur.x), y = Math.min(start.y, cur.y),
              w = Math.abs(cur.x - start.x), h = Math.abs(cur.y - start.y);
        ctx.fillStyle = 'rgba(255,204,56,0.12)'; ctx.fillRect(x, y, w, h);
        ctx.setLineDash([12, 8]);
        ctx.strokeStyle = '#ffcc38';
        ctx.shadowColor = 'rgba(0,0,0,0.5)';
        ctx.shadowBlur = 4;
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
        ctx.shadowBlur = 0;
    }
}

// ============ Export + Train ============
// Exporteert de huidige annotaties naar YOLO-indeling en start /api/train met FormData.
async function exportToYOLOAndTrain() {
    if (!state.items.length) { alert('Voeg eerst afbeeldingen toe.'); return; }

    // Train-parameters ophalen; className wordt YOLO's "names: [className]"
    const clsName = (els.className.value || 'object').trim() || 'object';
    const baseModelKey = els.modelSel?.value || null;
    const exportDir = (els.exportDir?.value || '').trim() || null;

    // Optionele hyperparams (alleen meesturen als ingevuld)
    const epochs      = (document.getElementById('epochs')?.value || '').trim();
    const batch       = (document.getElementById('batch')?.value || '').trim();
    const imgsz       = (document.getElementById('imgsz')?.value || '').trim();
    const lr0         = (document.getElementById('lr0')?.value || '').trim();
    const weightDecay = (document.getElementById('weightDecay')?.value || '').trim();
    const patience    = (document.getElementById('patience')?.value || '').trim();
    const augment     = document.getElementById('augment')?.checked ? 'true' : '';

    // Vanaf bestaand checkpoint hervatten?
    const resumeFrom = (els.resumeSel?.value || '').trim(); // leeg = start from base

    setStatus('Dataset opbouwen…', 'text-bg-warning');
    els.exportTrain.disabled = true;

    // Bouw dataset: we sturen IMG's + bijbehorende YOLO labelbestanden mee
    const fd = new FormData();
    let imgCount = 0, posCount = 0;

    for (const item of state.items) {
        if (!item.boxes.length) continue;

        // Render huidige canvas naar JPEG-blob (kwaliteit 0.95)
        const imgBlob = await new Promise(res => item.canvas.toBlob(res, "image/jpeg", 0.95));

        // Bestandsnaambasis (random) om pairs (jpg/txt) te groeperen
        const stem = crypto.randomUUID ? crypto.randomUUID() : (Math.random().toString(36).slice(2));

        // 1) Afbeelding uploaden onder images/<stem>.jpg
        fd.append('files', new File([imgBlob], `images/${stem}.jpg`));

        // 2) Labels in YOLO TXT-formaat (class_id cx cy w h) in genormaliseerde coördinaten
        const lines = [];
        for (const b of item.boxes.filter(b => b.label !== 'neg')) {
            // Alleen POSITIEVE boxes worden als traininglabels gebruikt (negatives = implicit/background)
            const cx = (b.x + b.w / 2) / item.w;
            const cy = (b.y + b.h / 2) / item.h;
            const ww = b.w / item.w;
            const hh = b.h / item.h;
            lines.push(`0 ${cx} ${cy} ${ww} ${hh}`); // '0' = class index (we trainen single-class op clsName)
            posCount++;
        }
        fd.append('files', new File([new Blob([lines.join('\n')], { type: 'text/plain' })], `labels/${stem}.txt`));
        imgCount++;
    }

    // Basisvalidatie: zonder positieve labels heeft trainen geen zin
    if (imgCount === 0 || posCount === 0) {
        setStatus('Geen (positieve) labels gevonden', 'text-bg-danger');
        els.exportTrain.disabled = false;
        return;
    }

    // Verplichte velden
    fd.append('class_name', clsName);
    if (baseModelKey) fd.append('base_model_key', baseModelKey);
    if (exportDir)    fd.append('export_dir', exportDir);

    // Optionele hyperparams – alleen meesturen als ingevuld
    if (epochs)      fd.append('epochs', epochs);
    if (batch)       fd.append('batch', batch);
    if (imgsz)       fd.append('imgsz', imgsz);
    if (lr0)         fd.append('lr0', lr0);
    if (weightDecay) fd.append('weight_decay', weightDecay);
    if (patience)    fd.append('patience', patience);
    if (augment)     fd.append('augment', 'true');
    if (resumeFrom)  fd.append('resume_from', resumeFrom);

    log(`Uploaden + trainen…  (resume: ${resumeFrom || 'nee'})`);
    setStatus('Uploaden…', 'text-bg-warning');

    // Start training job op de backend
    let resp;
    try {
        resp = await fetch('/api/train', { method: 'POST', body: fd });
    } catch (e) {
        setStatus('Netwerkfout bij upload', 'text-bg-danger');
        els.exportTrain.disabled = false;
        log(`Fout: ${e}`);
        return;
    }

    // Interpretatie van serverantwoord
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
        setStatus('Serverfout bij starten', 'text-bg-danger');
        log(`Fout: ${resp.status} ${JSON.stringify(data)}`);
        els.exportTrain.disabled = false;
        return;
    }

    // Job gestart: job_id tonen en polling aanzetten
    const jobId = data.job_id;
    state.jobId = jobId;
    els.jobInfo.textContent = jobId ? `Job: ${jobId}` : '';
    setStatus('Training gestart', 'text-bg-info');
    startPolling();
}

// Init: laad basismodellen en resume-keuzes zodra de pagina klaar is
(async function initTrainerUI() {
    await loadConfigAndModels();
})();


// ============ Polling ============
// Haal één keer de status/log op van de huidige training job en update UI.
async function pollOnce() {
    if (!state.jobId) return;
    try {
        const r = await fetch(`/api/train/${state.jobId}`);
        const info = await r.json();

        // Volledige log uit server tonen (server stuurt cumulatieve log)
        if (info.log) {
            els.log.textContent = info.log;
            els.log.scrollTop = els.log.scrollHeight;
        }

        // Statusafhandeling
        if (info.status === 'running') {
            setStatus('Trainen…', 'text-bg-info');

        } else if (info.status === 'done') {
            setStatus('Klaar ✔️ (model geladen)', 'text-bg-success');
            stopPolling();
            els.exportTrain.disabled = false;

            // Als de server een geëxporteerd modelpad heeft, toon een downloadknop
            if (info.export_path) {
                const url = `/api/train/${state.jobId}/download`;
                log(`Model geëxporteerd naar: ${info.export_path}`);

                const a = document.createElement('a');
                a.href = url;                     // veilige download endpoint
                a.className = 'btn btn-sm btn-outline-success mt-2';
                a.textContent = 'Download model (best.pt)';
                a.download = '';                  // hint aan browser

                els.log.append('\n');
                const wrap = document.createElement('div');
                wrap.appendChild(a);
                els.log.parentElement.appendChild(wrap);
            } else {
                // Geen export? Toon de run-directory of generieke "Klaar."
                log(info.run_dir ? `Run map: ${info.run_dir}` : 'Klaar.');
            }

        } else if (info.status === 'error') {
            setStatus('Fout bij training', 'text-bg-danger');
            stopPolling();
            els.exportTrain.disabled = false;

        } else {
            setStatus('Onbekende job', 'text-bg-secondary');
        }
    } catch (e) {
        setStatus('Poll-fout', 'text-bg-danger');
        log(`Poll error: ${e}`);
    }
}

// Polling helpers
function startPolling() {
    stopPolling();                     // eerst oude interval opkuisen
    els.cancelPoll.disabled = false;   // annuleer-knop active
    state.polling = setInterval(pollOnce, 2000); // elke 2s opnieuw
}
function stopPolling() {
    if (state.polling) {
        clearInterval(state.polling);
        state.polling = null;
    }
    els.cancelPoll.disabled = true;
}

// --- helpers om server-detecties in te tekenen ---
// Tekent resulterende detecties (from /api/detect) over de bestaande image.
function drawDetectionsOnItem(item, dets) {
    // Eerst alles resetten/overtekenen (zodat overlay schoon is)
    drawItem(item);

    const ctx = item.ctx;
    const cvs = item.canvas;

    ctx.lineWidth = Math.max(4, Math.round(cvs.width / 350));
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    // Verwacht dets: [{x1,y1,x2,y2,label,conf}, ...] met conf in [0..1]
    dets.forEach(d => {
        const x = d.x1, y = d.y1, w = d.x2 - d.x1, h = d.y2 - d.y1;

        // Groenachtige overlay voor detectie + slagschaduw
        ctx.fillStyle = 'rgba(52,211,153,0.18)';
        ctx.fillRect(x, y, w, h);
        ctx.shadowColor = 'rgba(0,0,0,0.7)';
        ctx.shadowBlur = 6;
        ctx.strokeStyle = '#22e38f';
        ctx.strokeRect(x, y, w, h);
        ctx.shadowBlur = 0;

        // Label met confidence (als badge-achtig blokje)
        const label = `${d.label} ${(d.conf * 100).toFixed(1)}%`;
        ctx.font = `${Math.max(12, Math.round((item.w) / 50))}px system-ui, sans-serif`;
        const padX = 8, padY = 5;
        const textW = ctx.measureText(label).width;
        const textH = parseInt(ctx.font, 10) + padY * 2;

        // Label boven de box plaatsen; zo niet, onderaan (als te weinig ruimte is)
        let lx = x, ly = y - textH - 2;
        if (ly < 0) ly = y + h + 2;
        if (lx + textW + padX * 2 > item.canvas.width) {
            lx = item.canvas.width - (textW + padX * 2) - 2;
        }

        // Donkere achtergrond + dunne outline + witte tekst
        ctx.fillStyle = 'rgba(0,0,0,0.85)';
        ctx.fillRect(lx, ly, textW + padX * 2, textH);
        ctx.strokeStyle = '#22e38f';
        ctx.lineWidth = 1;
        ctx.strokeRect(lx, ly, textW + padX * 2, textH);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, lx + padX, ly + textH - padY);
    });
}

// --- server detectie op één item ---
// Stuurt een JPEG-encode van het canvas naar /api/detect en tekent het resultaat in.
async function detectOnItemServer(item, conf, iou) {
    // Canvas → JPEG-blob
    const imgBlob = await new Promise(res => item.canvas.toBlob(res, "image/jpeg", 0.95));

    // FormData voor upload
    const fd = new FormData();
    fd.append('file', new File([imgBlob], `${item.name || 'image'}.jpg`));
    if (conf != null) fd.append('conf', String(conf));
    if (iou  != null) fd.append('iou',  String(iou));

    let resp;
    try {
        resp = await fetch('/api/detect', { method: 'POST', body: fd });
    } catch (e) {
        log(`Detect-fout (network): ${e}`);
        return;
    }
    if (!resp.ok) {
        const err = await resp.text();
        log(`Detect-fout (server): ${resp.status} ${err}`);
        return;
    }

    // Parse response en tekenen
    const data = await resp.json();
    const dets = Array.isArray(data.items) ? data.items : [];
    log(`Detecties (${item.name || 'image'}): ${dets.length}`);
    drawDetectionsOnItem(item, dets);
}

// --- detecteer alle kaarten ---
// Loopt over alle items, gebruikt UI thresholds, en roept detectOnItemServer.
async function detectAll() {
    // Lees thresholds uit de UI; fallback op redelijke defaults
    const conf = parseFloat((document.getElementById('detThresh')?.value || '').replace(',', '.')) || 0.5;
    const iou  = parseFloat((document.getElementById('detIou')?.value   || '').replace(',', '.')) || 0.45;

    setStatus('Detecteren…', 'text-bg-info');
    for (const it of state.items) {
        await detectOnItemServer(it, conf, iou);
    }
    setStatus('Gereed', 'text-bg-secondary');
}

// --- knoppen ---
// Detecteer alle
document.getElementById('detectAll')?.addEventListener('click', detectAll);

// "Kies bestanden" -> open file picker
els.pick.onclick = () => els.file.click();

// Wanneer gebruiker bestanden selecteert, maak kaarten aan
els.file.onchange = async (e) => {
    for (const f of e.target.files) {
        const url = URL.createObjectURL(f);
        addImageCard(f.name, url);
    }
};

// Voorbeeldafbeeldingen toevoegen (drie vaste samples)
els.sample.onclick = async () => {
    const urls = [
        "/static/samples/Screenshot from 2025-09-22 23-07-40.png",
        "/static/samples/Screenshot from 2025-09-22 23-08-08.png",
        "/static/samples/Screenshot from 2025-09-22 23-08-35.png",
    ];
    let i = 1;
    for (const u of urls) {
        addImageCard(`voorbeeld_${i++}.png`, u);
    }
};

// Alles leegmaken (UI en state)
els.clear.onclick = () => {
    els.thumbs.querySelectorAll('.col-12').forEach(card => card.remove());
    state.items = [];
    refreshKpis();
    log('Canvas geleegd.');
};

// Start export+train, en polling annuleren
els.exportTrain.onclick = exportToYOLOAndTrain;
els.cancelPoll.onclick  = stopPolling;

// Laad basismodellen + resume-opties vanuit backend
async function loadConfigAndModels() {
    try {
        // /api/config bevat onder meer "models" (beschikbare basismodellen) en huidige config
        const r = await fetch('/api/config');
        const data = await r.json();
        if (Array.isArray(data.models) && els.modelSel) {
            els.modelSel.innerHTML = '';
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m; opt.textContent = m;
                els.modelSel.appendChild(opt);
            });
            // Activeer huidig model in de dropdown
            if (data.config?.model_key && data.models.includes(data.config.model_key)) {
                els.modelSel.value = data.config.model_key;
            }
        }
    } catch (e) {
        log(`Kon modellen niet laden: ${e}`);
    }

    try {
        // /api/models geeft lijst van bestaande .pt bestanden (best/last/exported)
        const r2 = await fetch('/api/models');
        const items = await r2.json();
        // Eerste optie is "geen resume"
        els.resumeSel.innerHTML = `<option value="">(geen – start vanaf basismodel)</option>`;
        if (Array.isArray(items)) {
            items.forEach(it => {
                const opt = document.createElement('option');
                opt.value = it.path;   // volledig pad naar .pt
                opt.textContent = it.name; // nette naam (bv. "train7 / best.pt")
                els.resumeSel.appendChild(opt);
            });
        }
    } catch (e) {
        log(`Kon getrainde modellen niet ophalen: ${e}`);
    }
}

// Laad een gekozen checkpoint direct in de live viewer (zonder trainen)
// Handig om even snel te testen met /api/model/load
els.loadResume?.addEventListener('click', async () => {
    const path = els.resumeSel?.value || '';
    if (!path) { alert('Kies eerst een model in "Resume van bestaand model".'); return; }
    try {
        const r = await fetch('/api/model/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data?.error || r.status);
        log(`Model geladen: ${data.path}`);
    } catch (e) {
        log(`Model laden mislukt: ${e.message}`);
    }
});

// Initiele UI-stand
setStatus('Gereed', 'text-bg-secondary');
refreshKpis();

// === WebSocket om nieuwe smartphone-foto's live te ontvangen ===
// Verbindt met /ws en luistert enkel naar berichten van type "trainer_image".
// De backend stuurt deze na /api/trainer/upload-image (naam + URL).
(function initTrainerWS() {
    try {
        const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${wsProto}://${location.host}/ws`);

        ws.onopen  = () => log('WS verbonden (trainer_image updates).');
        ws.onclose = () => log('WS gesloten.');
        ws.onerror = (e) => log('WS fout: ' + (e?.message || ''));

        ws.onmessage = (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                if (msg?.type === 'trainer_image' && msg.url) {
                    // Direct kaart toevoegen in de annotatie-grid (zonder herladen)
                    addImageCard(msg.name || 'upload.jpg', msg.url);
                    log(`Nieuwe upload ontvangen: ${msg.name || msg.url}`, 'ok');
                }
            } catch (e) {
                // Niet-fatale JSON parse foutjes negeren
            }
        };
    } catch (e) {
        log('Kon WS niet openen: ' + e.message);
    }
})();
