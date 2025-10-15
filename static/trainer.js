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
  items: [],   // { name, imgEl, boxes:[{x,y,w,h,label}], masks:[[{x,y},...]], tempPoly:[], canvas, ctx, w, h }
  polling: null,
  jobId: null
};

// Snelkoppelingen naar DOM-elementen (eenmalig zoeken, later hergebruiken)
const els = {
  thumbs: document.getElementById('thumbs'),
  kPos: document.getElementById('kPos'),
  kNeg: document.getElementById('kNeg'),
  kImgs: document.getElementById('kImgs'),
  log: document.getElementById('log'),
  status: document.getElementById('status'),
  jobInfo: document.getElementById('jobInfo'),
  file: document.getElementById('file'),
  pick: document.getElementById('pick'),
  sample: document.getElementById('sample'),
  clear: document.getElementById('clear'),
  exportTrain: document.getElementById('exportTrain'),
  cancelPoll: document.getElementById('cancelPoll'),
  className: document.getElementById('className'),
  modelSel: document.getElementById('modelSel'),
  exportDir: document.getElementById('exportDir'),
  resumeSel: document.getElementById('resumeSel'),
  loadResume: document.getElementById('loadResumeBtn')
};

// ============ Tool mode ============
const tool = { mode: 'box' }; // 'box' | 'mask'
document.getElementById('toolBox')?.addEventListener('change', () => {
  tool.mode = 'box';
  const mt = document.getElementById('maskToolbar');
  if (mt) mt.style.display = 'none';
});
document.getElementById('toolMask')?.addEventListener('change', () => {
  tool.mode = 'mask';
  const mt = document.getElementById('maskToolbar');
  if (mt) mt.style.display = 'flex';
});


// ============ Helpers ============
function log(msg) { els.log.textContent += `\n${msg}`; els.log.scrollTop = els.log.scrollHeight; }
function setStatus(text, badge = 'text-bg-secondary') { els.status.className = 'badge ' + badge; els.status.textContent = text; }
function refreshKpis() {
  let p = 0, n = 0, imgs = state.items.length;
  state.items.forEach(it => {
    it.boxes.forEach(b => b.label === 'neg' ? n++ : p++);
    p += (it.masks?.length || 0);
  });
  els.kPos.textContent = p; els.kNeg.textContent = n; els.kImgs.textContent = imgs;
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
  const col = document.createElement('div'); col.className = 'col-12 col-md-6';
  const card = document.createElement('div'); card.className = 'image-card shadow-sm';

  const header = document.createElement('div'); header.className = 'd-flex justify-content-between align-items-center p-2 border-bottom bg-light';
  header.innerHTML = `<strong class="text-truncate">${fileName}</strong>`;

  const canvasWrap = document.createElement('div'); canvasWrap.className = 'canvas-wrap';
  const img = new Image(); img.src = imgUrl; img.crossOrigin = 'anonymous';
  const canvas = document.createElement('canvas'); const ctx = canvas.getContext('2d');

  canvasWrap.appendChild(img); canvasWrap.appendChild(canvas);
  card.appendChild(header); card.appendChild(canvasWrap);
  col.appendChild(card);
  els.thumbs.prepend(col);

  const item = { name: fileName, imgEl: img, boxes: [], masks: [], tempPoly: [], canvas, ctx, w: 0, h: 0 };
  state.items.push(item);

  img.onload = () => {
    item.w = img.naturalWidth; item.h = img.naturalHeight;
    canvas.width = item.w; canvas.height = item.h;
    drawItem(item); refreshKpis();
  };

  let drawing = false, start = null, cur = null;

  function toNatural(e) {
    const r = canvas.getBoundingClientRect();
    const cx = (e.clientX - r.left) * canvas.width / r.width;
    const cy = (e.clientY - r.top) * canvas.height / r.height;
    return { x: cx, y: cy };
  }

  // BOX: drag; MASK: click-points
  canvas.addEventListener('mousedown', e => {
    const p = toNatural(e);
    if (tool.mode === 'box') {
      drawing = true; start = p; cur = null;
    } else {
      item.tempPoly.push(p);
      drawItem(item);
    }
  });

  canvas.addEventListener('mousemove', e => {
    if (tool.mode !== 'box' || !drawing) return;
    cur = toNatural(e);
    drawItem(item, start, cur);
  });

  window.addEventListener('mouseup', e => {
    if (tool.mode !== 'box' || !drawing) return;
    drawing = false;
    const end = toNatural(e);
    const x = Math.min(start.x, end.x), y = Math.min(start.y, end.y);
    const w = Math.abs(end.x - start.x), h = Math.abs(end.y - start.y);
    if (w > 10 && h > 10) {
      const isNeg = e.shiftKey;
      item.boxes.push({ x, y, w, h, label: isNeg ? 'neg' : 'pos' });
      drawItem(item); refreshKpis();
      log(`${isNeg ? '- Negatieve' : '+ Positieve'} box toegevoegd (${fileName})`);
    }
  });

  // Dblclick = mask sluiten
  canvas.addEventListener('dblclick', () => {
    if (tool.mode !== 'mask' || item.tempPoly.length < 3) return;
    item.masks.push(item.tempPoly.slice());
    item.tempPoly = [];
    drawItem(item); refreshKpis(); log(`+ Mask toegevoegd (${fileName})`);
  });

  // Rechtermuisklik = undo
  canvas.addEventListener('contextmenu', e => {
    e.preventDefault();
    if (tool.mode === 'mask') {
      if (item.tempPoly.length) item.tempPoly.pop();
      else if (item.masks.length) { item.masks.pop(); log('− Laatste mask verwijderd (' + fileName + ')'); }
    } else {
      if (item.boxes.length) { item.boxes.pop(); log('− Laatste box verwijderd (' + fileName + ')'); }
    }
    drawItem(item); refreshKpis();
  });

  return item;
}

document.getElementById('maskUndoPoint')?.addEventListener('click', () => {
  // undo geldt op laatst aangemaakte/gewijzigde card; hier simpel: op elk item tempPoly proberen
  for (const it of state.items) {
    if (it.tempPoly?.length) {
      it.tempPoly.pop();
      drawItem(it);
      break;
    }
  }
});

document.getElementById('maskClose')?.addEventListener('click', () => {
  for (const it of state.items) {
    if (it.tempPoly?.length >= 3) {
      it.masks.push(it.tempPoly.slice());
      it.tempPoly = [];
      drawItem(it);
      refreshKpis();
      log('+ Mask toegevoegd (' + it.name + ')');
      break;
    }
  }
});

document.getElementById('maskUndoPoly')?.addEventListener('click', () => {
  for (const it of state.items) {
    if (it.masks?.length) {
      it.masks.pop();
      drawItem(it);
      refreshKpis();
      log('− Laatste mask verwijderd (' + it.name + ')');
      break;
    }
  }
});

// ============ Tekenen ============
// Teken image + alle definitieve boxes. Optioneel ook een "ghost" selectie tijdens het slepen.
function drawItem(item, start = null, cur = null) {
  const { ctx, canvas, imgEl } = item;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgEl, 0, 0, item.w, item.h);

  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  // Definitieve MASKS (groen gevuld + rand)
  for (const poly of item.masks) {
    if (!poly?.length) continue;
    ctx.beginPath();
    ctx.moveTo(poly[0].x, poly[0].y);
    for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x, poly[i].y);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,255,195,0.18)';
    ctx.strokeStyle = '#00ffc3';
    ctx.lineWidth = Math.max(3, Math.round(canvas.width / 400));
    ctx.fill();
    ctx.stroke();
  }

  // Tijdelijke poly (mask in aanbouw)
  if (item.tempPoly?.length) {
    const tp = item.tempPoly;
    ctx.beginPath();
    ctx.moveTo(tp[0].x, tp[0].y);
    for (let i = 1; i < tp.length; i++) ctx.lineTo(tp[i].x, tp[i].y);
    ctx.strokeStyle = '#ffcc38';
    ctx.setLineDash([10, 6]);
    ctx.lineWidth = Math.max(3, Math.round(canvas.width / 400));
    ctx.stroke();
    ctx.setLineDash([]);

    // punten
    ctx.fillStyle = '#ffcc38';
    for (const p of tp) {
      ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(3, canvas.width / 300), 0, Math.PI * 2); ctx.fill();
    }
  }

  // BOXES (pos = groen, neg = rood)
  for (const b of item.boxes) {
    const x = b.x, y = b.y, w = b.w, h = b.h;
    ctx.fillStyle = (b.label === 'neg') ? 'rgba(239,68,68,0.15)' : 'rgba(52,211,153,0.18)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = (b.label === 'neg') ? '#ef4444' : '#22e38f';
    ctx.lineWidth = Math.max(3, Math.round(canvas.width / 400));
    ctx.strokeRect(x, y, w, h);
  }

  // Ghost box tijdens slepen (alleen box-tool)
  if (start && cur) {
    const x = Math.min(start.x, cur.x), y = Math.min(start.y, cur.y),
      w = Math.abs(cur.x - start.x), h = Math.abs(cur.y - start.y);
    ctx.fillStyle = 'rgba(255,204,56,0.12)'; ctx.fillRect(x, y, w, h);
    ctx.setLineDash([12, 8]);
    ctx.strokeStyle = '#ffcc38';
    ctx.lineWidth = Math.max(3, Math.round(canvas.width / 400));
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}


// ============ Export + Train ============
// Exporteert de huidige annotaties naar YOLO-indeling en start /api/train met FormData.
async function exportToYOLOAndTrain() {
  if (!state.items.length) { alert('Voeg eerst afbeeldingen toe.'); return; }

  const clsName = (els.className.value || 'object').trim() || 'object';
  const baseModelKeyRaw = els.modelSel?.value || null;
  const exportDir = (els.exportDir?.value || '').trim() || null;

  const epochs = (document.getElementById('epochs')?.value || '').trim();
  const batch = (document.getElementById('batch')?.value || '').trim();
  const imgsz = (document.getElementById('imgsz')?.value || '').trim();
  const lr0 = (document.getElementById('lr0')?.value || '').trim();
  const weightDecay = (document.getElementById('weightDecay')?.value || '').trim();
  const patience = (document.getElementById('patience')?.value || '').trim();
  const augment = document.getElementById('augment')?.checked ? 'true' : '';
  const resumeFrom = (els.resumeSel?.value || '').trim();

  setStatus('Dataset opbouwen…', 'text-bg-warning');
  els.exportTrain.disabled = true;

  const fd = new FormData();
  let imgCount = 0, posCount = 0, usedSegMasks = 0;

  for (const item of state.items) {
    const hasAny = item.boxes.some(b => b.label !== 'neg') || (item.masks?.length);
    if (!hasAny) continue;

    // Export ZONDER overlays: render bronafbeelding naar offscreen canvas
    const off = document.createElement('canvas'); off.width = item.w; off.height = item.h;
    const octx = off.getContext('2d'); octx.drawImage(item.imgEl, 0, 0, item.w, item.h);
    const imgBlob = await new Promise(res => off.toBlob(res, "image/jpeg", 0.95));

    const stem = (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2));
    fd.append('files', new File([imgBlob], `images/${stem}.jpg`));

    const lines = [];

    // Masks -> YOLOv8-seg
    if (Array.isArray(item.masks)) {
      for (const poly of item.masks) {
        if (!poly || poly.length < 3) continue;
        const coords = [];
        for (const p of poly) {
          const xn = Math.max(0, Math.min(1, p.x / item.w));
          const yn = Math.max(0, Math.min(1, p.y / item.h));
          coords.push(xn.toFixed(6), yn.toFixed(6));
        }
        lines.push(`0 ${coords.join(' ')}`); // single-class
        usedSegMasks++;
      }
    }

    // Box-only (alleen als er geen mask is voor dit beeld)
    if (!item.masks?.length) {
      for (const b of item.boxes.filter(b => b.label !== 'neg')) {
        const cx = (b.x + b.w / 2) / item.w, cy = (b.y + b.h / 2) / item.h;
        const ww = b.w / item.w, hh = b.h / item.h;
        lines.push(`0 ${cx} ${cy} ${ww} ${hh}`);
        posCount++;
      }
    }

    fd.append('files', new File([new Blob([lines.join('\n')], { type: 'text/plain' })], `labels/${stem}.txt`));
    imgCount++;
  }

  if (imgCount === 0 || (posCount === 0 && usedSegMasks === 0)) {
    setStatus('Geen (positieve) labels of masks gevonden', 'text-bg-danger');
    els.exportTrain.disabled = false;
    return;
  }

  // Verplichte & optionele velden
  fd.append('class_name', clsName);
  if (exportDir) fd.append('export_dir', exportDir);
  if (epochs) fd.append('epochs', epochs);
  if (batch) fd.append('batch', batch);
  if (imgsz) fd.append('imgsz', imgsz);
  if (lr0) fd.append('lr0', lr0);
  if (weightDecay) fd.append('weight_decay', weightDecay);
  if (patience) fd.append('patience', patience);
  if (augment) fd.append('augment', 'true');
  if (resumeFrom) fd.append('resume_from', resumeFrom);

  // Kies basismodel, switch naar -seg als er masks zijn
  if (baseModelKeyRaw) {
    const baseModelKey = (usedSegMasks > 0 && !/-seg$/.test(baseModelKeyRaw))
      ? baseModelKeyRaw.replace(/(yolov8[nsmlx])$/, '$1-seg')
      : baseModelKeyRaw;
    fd.append('base_model_key', baseModelKey);
  }

  log(`Uploaden + trainen…  (resume: ${resumeFrom || 'nee'})`);
  setStatus('Uploaden…', 'text-bg-warning');

  let resp;
  try { resp = await fetch('/api/train', { method: 'POST', body: fd }); }
  catch (e) { setStatus('Netwerkfout bij upload', 'text-bg-danger'); els.exportTrain.disabled = false; log(`Fout: ${e}`); return; }

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    setStatus('Serverfout bij starten', 'text-bg-danger');
    log(`Fout: ${resp.status} ${JSON.stringify(data)}`);
    els.exportTrain.disabled = false; return;
  }

  state.jobId = data.job_id;
  els.jobInfo.textContent = state.jobId ? `Job: ${state.jobId}` : '';
  setStatus('Training gestart', 'text-bg-info');
  startPolling();
}


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
  // Achtergrond (bron + bestaande annotaties) opnieuw tekenen
  drawItem(item);

  const ctx = item.ctx;
  const cvs = item.canvas;
  const line = Math.max(3, Math.round(cvs.width / 400));
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  dets.forEach(d => {
    const label = `${d.label} ${(d.conf * 100).toFixed(1)}%`;

    if (tool.mode === 'mask' && Array.isArray(d.mask) && d.mask.length >= 3) {
      // === MASK tekenen ===
      ctx.beginPath();
      ctx.moveTo(d.mask[0][0], d.mask[0][1]);
      for (let i = 1; i < d.mask.length; i++) {
        ctx.lineTo(d.mask[i][0], d.mask[i][1]);
      }
      ctx.closePath();

      ctx.fillStyle = 'rgba(0,255,195,0.18)';
      ctx.strokeStyle = '#00ffc3';
      ctx.lineWidth = line;
      ctx.fill();
      ctx.stroke();

      // label nabij eerste punt
      ctx.font = `${Math.max(12, Math.round(item.w / 50))}px system-ui, sans-serif`;
      const tx = d.mask[0][0] + 6;
      const ty = d.mask[0][1] + 18;
      const padX = 8, padY = 5;
      const textW = ctx.measureText(label).width;
      const textH = parseInt(ctx.font, 10) + padY * 2;

      ctx.fillStyle = 'rgba(0,0,0,0.85)';
      ctx.fillRect(tx - padX, ty - textH + padY, textW + padX * 2, textH);
      ctx.strokeStyle = '#00ffc3';
      ctx.lineWidth = 1;
      ctx.strokeRect(tx - padX, ty - textH + padY, textW + padX * 2, textH);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, tx, ty);

    } else {
      // === BOX tekenen (default) ===
      const x = d.x1, y = d.y1, w = d.x2 - d.x1, h = d.y2 - d.y1;

      ctx.fillStyle = 'rgba(52,211,153,0.18)';
      ctx.strokeStyle = '#22e38f';
      ctx.lineWidth = line;

      ctx.fillRect(x, y, w, h);
      ctx.shadowColor = 'rgba(0,0,0,0.7)';
      ctx.shadowBlur = 6;
      ctx.strokeRect(x, y, w, h);
      ctx.shadowBlur = 0;

      ctx.font = `${Math.max(12, Math.round(item.w / 50))}px system-ui, sans-serif`;
      const padX = 8, padY = 5;
      const textW = ctx.measureText(label).width;
      const textH = parseInt(ctx.font, 10) + padY * 2;

      let lx = x, ly = y - textH - 2;
      if (ly < 0) ly = y + h + 2;
      if (lx + textW + padX * 2 > item.canvas.width) {
        lx = item.canvas.width - (textW + padX * 2) - 2;
      }

      ctx.fillStyle = 'rgba(0,0,0,0.85)';
      ctx.fillRect(lx, ly, textW + padX * 2, textH);
      ctx.strokeStyle = '#22e38f';
      ctx.lineWidth = 1;
      ctx.strokeRect(lx, ly, textW + padX * 2, textH);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, lx + padX, ly + textH - padY);
    }
  });
}

// --- server detectie op één item ---
// Stuurt een JPEG-encode van het canvas naar /api/detect en tekent het resultaat in.
async function detectOnItemServer(item, conf, iou) {
  // exporteer de kale bronafbeelding (geen overlays)
  const off = document.createElement('canvas');
  off.width = item.w; off.height = item.h;
  const octx = off.getContext('2d');
  octx.drawImage(item.imgEl, 0, 0, item.w, item.h);
  const imgBlob = await new Promise(res => off.toBlob(res, "image/jpeg", 0.95));

  const fd = new FormData();
  fd.append('file', new File([imgBlob], `${item.name || 'image'}.jpg`));
  if (conf != null) fd.append('conf', String(conf));
  if (iou != null) fd.append('iou', String(iou));

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
  const iou = parseFloat((document.getElementById('detIou')?.value || '').replace(',', '.')) || 0.45;

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
    "/static/samples/Screenshot from 2025-09-24 01-39-04.png",
    "/static/samples/Screenshot from 2025-09-24 01-39-30.png",
    "/static/samples/Screenshot from 2025-09-24 01-39-48.png",
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
els.cancelPoll.onclick = stopPolling;

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

(async function initTrainerUI() {
  await loadConfigAndModels();
})();


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

    ws.onopen = () => log('WS verbonden (trainer_image updates).');
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
