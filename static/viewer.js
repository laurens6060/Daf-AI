(function () {
    // ============================================================
    //  UI / Viewer-side logica voor de Realtime AI-detectie demo
    //  - Verbindt met WebSocket voor live 'config' & 'detections'
    //  - Rendered tabellen/badges (Bootstrap) voor items & counts
    //  - Stuurt instellingen (conf, iou, imgsz, stabilisatie) naar backend
    //  - Laadt basismodellen & getrainde modellen + active/primary badges
    //  - Leest Collections & Products (read-only) en toont properties
    // ============================================================

    // ===== DOM referenties (alleen ophalen 1x, later hergebruiken) =====
    const tbody = document.getElementById("detBody");         // tabel-body voor individuele detecties
    const stickyBox = document.getElementById("stickyBox");   // compact overzicht (badges) met "present counts"

    // Slider/inputs voor detector-parameters
    const confInp = document.getElementById("confInp"); const confVal = document.getElementById("confVal");
    const iouInp = document.getElementById("iouInp"); const iouVal = document.getElementById("iouVal");
    const imgszInp = document.getElementById("imgszInp"); const imgszVal = document.getElementById("imgszVal");

    // Stabilisatie-parameters (tracking smoothing / hysterese)
    const holdInp = document.getElementById("holdInp"); const holdVal = document.getElementById("holdVal");
    const hitsInp = document.getElementById("hitsInp"); const hitsVal = document.getElementById("hitsVal");
    const emaInp = document.getElementById("emaInp"); const emaVal = document.getElementById("emaVal");

    const classesBox = document.getElementById("classesBox"); // container voor dynamisch gegenereerde class-checkboxen
    const saveBtn = document.getElementById("saveBtn");       // knop om conf/iou/imgsz/allowed_classes op te slaan
    const allowAllBtn = document.getElementById("allowAllBtn");
    const clearAllBtn = document.getElementById("clearAllBtn");
    const saveStabBtn = document.getElementById("saveStab");  // knop om hold_ms/min_hits/ema_alpha op te slaan

    // Modellen-sectie (basismodellen radio / getrainde modellen checkboxen)
    const baseModelsBox = document.getElementById('baseModelsBox');
    const trainedModelsBox = document.getElementById('trainedModelsBox');
    const applyBaseModelBtn = document.getElementById('applyBaseModelBtn');
    const applyTrainedModelBtn = document.getElementById('applyTrainedModelBtn');

    // Collections & Products (read-only lijst in viewer)
    const collectionsList = document.getElementById('collectionsList');
    const productsBody = document.getElementById('productsBody');
    const productsCount = document.getElementById('productsCount');

    const showMasksInp = document.getElementById("showMasksInp");
    const showBoxesInp = document.getElementById("showBoxesInp");
    const showContoursInp = document.getElementById("showContoursInp");

    const contoursBox = document.getElementById('contoursBox');
    const applyContourBtn = document.getElementById('applyContourBtn');
    const disableContourBtn = document.getElementById('disableContourBtn');

    const overlay = document.getElementById('overlay');
    const video = document.getElementById('video');

    const captureBtn = document.getElementById('captureBtn');
    const captureStatus = document.getElementById('captureStatus');
    const photoRadios = document.querySelectorAll('input[name="photoMode"]');

    let autoPhotoMode = false;
    let lastAutoCapture = 0;
    const AUTO_CAPTURE_INTERVAL_MS = 4000; // minstens 4s tussen foto's

    photoRadios.forEach(r => {
        r.addEventListener('change', () => {
            autoPhotoMode = (document.getElementById('photoAuto').checked);
            captureStatus.textContent = autoPhotoMode
                ? 'Automatische modus actief: foto wordt genomen bij centrering.'
                : 'Handmatige modus.';
        });
    });

    async function takePhoto() {
        try {
            captureStatus.textContent = 'Bezig met foto-opname...';
            // Overlay tijdelijk verbergen
            overlay.style.display = 'none';
            const resp = await fetch('/api/capture', { method: 'POST' });
            overlay.style.display = ''; // overlay terug tonen

            if (!resp.ok) throw new Error(await resp.text());
            const data = await resp.json().catch(() => ({}));
            captureStatus.textContent = data?.path
                ? `Foto opgeslagen: ${data.path}`
                : 'Foto genomen.';
        } catch (e) {
            captureStatus.textContent = 'Fout bij foto-opname: ' + e.message;
        }
    }

    captureBtn?.addEventListener('click', takePhoto);


    const productMatchBoxId = 'productMatchBox';
    let stableCenterStart = null;
    let lastCenteredProductId = null;
    const CENTER_TOLERANCE = 0.2;   // 20% marge van scherm
    const STABILITY_MS = 400;       // 0.4 seconde stabiel vereist



    showMasksInp?.addEventListener('change', () => {
        POSTJSON('/api/config', { show_masks: !!showMasksInp.checked })
            .catch(e => alert('Kon show_masks niet opslaan: ' + e.message));
    });

    showBoxesInp?.addEventListener('change', () => {
        POSTJSON('/api/config', { show_boxes: !!showBoxesInp.checked })
            .catch(e => alert('Kon show_boxes niet opslaan: ' + e.message));
    });

    showContoursInp?.addEventListener('change', () => {
        POSTJSON('/api/config', { contour_match_enabled: !!showContoursInp.checked })
            .catch(e => alert('Kon contour instelling niet opslaan: ' + e.message));
    });


    // Wanneer de modal openklapt, bouw de QR naar /sender op dezelfde host/poort/protocol
    const qrModalEl = document.getElementById('qrModal');

    // ===== In-memory staat =====
    let currentConfig = null; // laatste 'config' van de server (als bron van waarheid)
    let presentMap = {};      // snelle lookup voor label -> count (optioneel voor toekomstige UI-logica)

    let collections = [];           // laatste geladen collections
    let products = [];              // laatste geladen producten (eventueel gefilterd op collection)
    let selectedCollectionId = null;

    let lastMatchedProductIds = new Set();


    // ===== Kleine helpers =====

    function productNameById(pid) {
        const p = (products || []).find(x => String(x.id) === String(pid));
        return p ? String(p.name || '') : null;
    }

    /**
     * Controleer of de herkende contour of product in het midden van het beeld ligt.
     * msg moet frame_w/h + eventueel cx/cy bevatten.
     */
    function isCentered(msg) {
        if (!msg || typeof msg.frame_w !== 'number' || typeof msg.frame_h !== 'number') return false;
        const cx = Number(msg.cx ?? msg.center_x ?? msg.contour_cx ?? msg.frame_w / 2);
        const cy = Number(msg.cy ?? msg.center_y ?? msg.contour_cy ?? msg.frame_h / 2);
        const tolX = msg.frame_w * CENTER_TOLERANCE;
        const tolY = msg.frame_h * CENTER_TOLERANCE;
        const centerX = msg.frame_w / 2;
        const centerY = msg.frame_h / 2;
        return Math.abs(cx - centerX) <= tolX && Math.abs(cy - centerY) <= tolY;
    }


    /**
     * Toon de best/herkende productnaam (1e uit matched_product_ids) als badge
     * onder de stickyBox (“In beeld”).
     */
    function renderTopMatchedProduct(matchedIds, statusText) {
        if (!stickyBox) return;
        let box = document.getElementById(productMatchBoxId);
        if (!box) {
            box = document.createElement('div');
            box.id = productMatchBoxId;
            box.className = 'mt-2';
            stickyBox.insertAdjacentElement('beforeend', box);
        }

        if (statusText) {
            box.innerHTML = `<div class="small text-muted">${esc(statusText)}</div>`;
            return;
        }

        const arr = Array.isArray(matchedIds) ? matchedIds.map(String) : [];
        if (!arr.length) {
            box.innerHTML = '';
            return;
        }

        const topId = arr[0];
        const topName = productNameById(topId) || topId;

        box.innerHTML = `
      <div class="small text-muted">Herkenning (gecentreerd):</div>
      <span class="badge text-bg-success">${esc(topName)}</span>
    `;
    }



    /**
     * Koppelt een <input type="range"> aan een label element en houdt
     * de tekst weergave synchroon met de sliderwaarde.
     * @param {HTMLInputElement} inp - slider
     * @param {HTMLElement} label - label waar de zichtbare waarde in komt
     * @param {(v:string)=>string} fmt - formatter voor labeltekst
     */
    const bindRange = (inp, label, fmt = (v) => v) => {
        if (!inp || !label) return;      // defensief: alleen binden als beide bestaan
        const update = () => label.textContent = fmt(inp.value);
        inp.addEventListener("input", update); // live updaten tijdens schuiven
        update();                                // initieel meteen goed zetten
    };

    // Slider-labels netjes formatteren
    bindRange(confInp, confVal, v => (+v).toFixed(2));
    bindRange(iouInp, iouVal, v => (+v).toFixed(2));
    bindRange(imgszInp, imgszVal, v => `${v}`);
    bindRange(holdInp, holdVal, v => `${v} ms`);
    bindRange(hitsInp, hitsVal, v => `${v}`);
    bindRange(emaInp, emaVal, v => (+v).toFixed(2));

    if (!qrModalEl) return;

    const senderUrl = () => {
        // Gebruik exact dezelfde origin als de viewer (juiste IP/poort/https):
        const origin = window.location.origin;     // bv. http://192.168.1.50:8000
        return origin + "/sender";
    };

    const qrContainer = document.getElementById('qrTarget');
    const qrLinkEl = document.getElementById('qrLink');
    const copyBtn = document.getElementById('copyLinkBtn');

    let qr;

    qrModalEl.addEventListener('show.bs.modal', function () {
        const url = senderUrl();
        qrLinkEl.textContent = url;
        qrLinkEl.href = url;

        qrContainer.innerHTML = "";

        if (window.QRCode) {
            new QRCode(qrContainer, {
                text: url,
                width: 220,
                height: 220,
                correctLevel: QRCode.CorrectLevel.M
            });
        } else {
            // Fallback: duidelijke melding
            const warn = document.createElement('div');
            warn.className = 'text-danger small';
            warn.textContent = 'QR-module niet geladen; klik op de link hierboven.';
            qrContainer.appendChild(warn);
        }
    });


    copyBtn?.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(senderUrl());
            copyBtn.textContent = "Gekopieerd!";
            setTimeout(() => copyBtn.textContent = "Kopieer link", 1200);
        } catch (e) {
            alert("Kopiëren mislukt, kopieer handmatig: " + senderUrl());
        }
    });

    /**
     * GET helper met JSON-parsing + error op HTTP status != 2xx.
     * Gebruik: const data = await GET('/api/config');
     */
    async function GET(url) {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`${r.status} ${url}`);
        return r.json();
    }

    /**
     * POST helper voor JSON body + JSON response. Gooit Error met server 'error' indien aanwezig.
     */
    async function POSTJSON(url, body) {
        const r = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data?.error || r.statusText);
        return data;
    }

    /**
     * HTML escapefunctie om XSS via labels/waarden te voorkomen
     */
    const esc = s => String(s).replace(/[&<>"']/g, m => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }[m]));

    // --- normalisatie & enkelvoud/meervoud varianten ---
    const norm = s => String(s || '').trim().toLowerCase();

    function nameVariants(name) {
        const n = norm(name);
        const vars = new Set([n]);

        // simpel meervoud/enkelvoud
        if (n.endsWith('ies')) vars.add(n.slice(0, -3) + 'y');     // batteries -> battery
        if (n.endsWith('y')) vars.add(n.slice(0, -1) + 'ies');   // battery -> batteries
        if (n.endsWith('es')) vars.add(n.slice(0, -2));           // boxes -> box
        if (n.endsWith('s')) vars.add(n.slice(0, -1));           // earbuds -> earbud
        else vars.add(n + 's');                  // earbud -> earbuds

        return Array.from(vars);
    }

    function flattenMarkers(data = []) {
        const out = [];
        for (const m of (Array.isArray(data) ? data : [])) {
            if (m && Array.isArray(m.items)) {
                for (const it of m.items) {
                    out.push({
                        ...it,
                        // neem een bruikbaar label mee
                        label: it.label ?? it.expected_label ?? m.name ?? m.label,
                    });
                }
            } else {
                out.push(m);
            }
        }
        return out;
    }


    function drawPOIOverlay(markers = [], meta = {}) {
        console.debug('poi(flat):', markers);
        const ctx = overlay.getContext('2d');

        // Canvas matcht zichtbare maat van het beeld
        const rect = video.getBoundingClientRect();
        overlay.width = Math.max(1, Math.round(rect.width));
        overlay.height = Math.max(1, Math.round(rect.height));
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        if (!markers.length) return;

        // Frame-afmetingen komen uit meta (niet uit msg!)
        const frameW = Number(meta.frameW ?? meta.frame_w ?? meta.w) || overlay.width;
        const frameH = Number(meta.frameH ?? meta.frame_h ?? meta.h) || overlay.height;

        const scaleX = overlay.width / frameW;
        const scaleY = overlay.height / frameH;

        // helper om meerdere sleutel-namen te accepteren
        const val = (obj, ...keys) => {
            for (const k of keys) if (obj[k] != null) return obj[k];
            return undefined;
        };

        markers.forEach(m => {
            let x = val(m, 'x', 'cx', 'x01');
            let y = val(m, 'y', 'cy', 'y01');
            if (x == null || y == null) return;

            let r = val(m, 'r', 'radius', 'radius01');
            if (r == null) r = 12;

            const looksNormalized = x <= 1 && y <= 1 && r <= 1.5;
            if (looksNormalized) {
                x *= frameW;
                y *= frameH;
                r *= Math.min(frameW, frameH);
            }

            const dx = x * scaleX;
            const dy = y * scaleY;
            const dr = Math.max(6, r * ((scaleX + scaleY) / 2));

            ctx.beginPath();
            ctx.arc(dx, dy, dr, 0, Math.PI * 2);
            ctx.lineWidth = 3;
            ctx.strokeStyle = (m.ok ?? true) ? '#19a974' : '#e03131';
            ctx.stroke();

            const labelText = `${val(m, 'found', 'count') ?? 0}/${val(m, 'required', 'req') ?? 1} ${val(m, 'label', 'expected_label') ?? ''}`;
            ctx.font = '12px system-ui';
            ctx.fillStyle = (m.ok ?? true) ? '#19a974' : '#e03131';
            ctx.fillText(labelText, dx + 8, dy - 8);
        });
    }




    // haal count op uit presentMap met tolerant matching
    function getActualCountFor(name, pmap) {
        const keys = Object.keys(pmap || {});
        const wanted = nameVariants(name);
        // 1) exacte match op varianten
        for (const v of wanted) {
            if (pmap.hasOwnProperty(v)) return Number(pmap[v]) || 0;
        }
        // 2) fallback: vind een key met zelfde variant (robuster bij kleine variaties)
        for (const k of keys) {
            const kv = nameVariants(k);
            if (kv.some(v => wanted.includes(v))) return Number(pmap[k]) || 0;
        }
        return 0;
    }

    // onthoud wat we al verstuurd hebben (debounce per product)
    const lastRejectKeyByPid = new Map();

    function detailedMismatches(prod, pmap) {
        if (!prod || !prod.properties) return [];
        const out = [];
        for (const [name, expectedRaw] of Object.entries(prod.properties)) {
            const expected = Number(expectedRaw);
            if (!Number.isFinite(expected) || expected === 0) continue;
            const actual = getActualCountFor(name, pmap);
            if (actual !== expected) out.push({ name, expected, actual });
        }
        return out;
    }


    function computePropertyCheck(prod, presentMap) {
        const res = { mismatches: [], oks: [] };
        if (!prod || !prod.properties || typeof prod.properties !== 'object') return res;

        for (const [name, expectedRaw] of Object.entries(prod.properties)) {
            const expected = Number(expectedRaw);
            if (!Number.isFinite(expected) || expected === 0) continue; // 0 = geen check

            const actual = getActualCountFor(name, presentMap); // ← tolerant
            if (actual !== expected) {
                res.mismatches.push(`Productiefout: ziet ${actual} ${name}, moet ${expected} ${name} zien.`);
            } else {
                res.oks.push(`Zoekt ${expected} ${name}, ziet ${actual} ${name}.`);
                // res.oks.push(`Zoekt ${expected} aantal ${name}, ziet ${actual} aantal ${name}.`);
            }
        }
        return res;
    }

    async function loadContoursList() {
        if (!contoursBox) return;
        contoursBox.innerHTML = '<div class="text-muted">Laden…</div>';
        try {
            const items = await GET('/api/contours');
            if (!items.length) {
                contoursBox.innerHTML = '<div class="text-muted">Geen contouren</div>';
                return;
            }
            contoursBox.innerHTML = items.map(c => `
      <label class="list-group-item d-flex align-items-center justify-content-between">
        <span>${esc(c.name)} <span class="text-muted">(${esc(c.type_key || '')})</span></span>
        <input class="form-check-input" type="checkbox" name="contourChk" value="${esc(c.id)}">
      </label>
    `).join('');
        } catch (e) {
            contoursBox.innerHTML = `<div class="text-danger">Contourlijst laden mislukt</div>`;
        }
    }

    applyContourBtn?.addEventListener('click', async () => {
        const ids = Array.from(document.querySelectorAll('input[name="contourChk"]:checked'))
            .map(el => el.value);
        try {
            await POSTJSON('/api/config', {
                contour_match_enabled: true,
                active_contour_ids: ids  // [] = geen; lijst = subset; (weg laten of null = alle)
            });
        } catch (e) {
            alert('Kon contour selectie toepassen: ' + e.message);
        }
    });

    disableContourBtn?.addEventListener('click', async () => {
        try {
            await POSTJSON('/api/config', {
                contour_match_enabled: false,
                active_contour_ids: []  // expliciet leeg (geen contouren actief)
            });
        } catch (e) {
            alert('Kon contour uitschakelen: ' + e.message);
        }
    });



    /**
     * Plaats (of verwijder) een alert-rij direct onder de product rij in de tabel.
     */
    function renderProductAlertRow(pid, check) {
        const tr = document.querySelector(
            `#productsBody tr.product-row[data-pid="${CSS.escape(String(pid))}"]`
        );
        if (!tr) return;

        // oude alert-rij weg
        const next = tr.nextElementSibling;
        if (next && next.matches('.product-alert-row') && next.dataset.pid === String(pid)) {
            next.remove();
        }

        // niets te tonen (geen properties met >0)
        if (!check || (check.mismatches.length === 0 && check.oks.length === 0)) return;

        const isOk = check.mismatches.length === 0;
        const messages = isOk ? check.oks : check.mismatches;
        const cls = isOk ? 'alert-success' : 'alert-danger';

        const alertTr = document.createElement('tr');
        alertTr.className = 'product-alert-row';
        alertTr.dataset.pid = String(pid);
        alertTr.innerHTML = `
    <td colspan="2" class="product-alert">
      <div class="alert ${cls}" role="alert">
        ${messages.map(m => `<div>${m}</div>`).join('')}
      </div>
    </td>
  `;
        tr.insertAdjacentElement('afterend', alertTr);
    }


    /**
     * Voor alle gematchte producten: (her)bereken en toon de alerts.
     * Voor niet-gematchte producten: alerts weghalen.
     */
    function updateAllProductAlerts(matchedIds) {
        const idSet = new Set((matchedIds || []).map(String));

        document.querySelectorAll('#productsBody tr.product-row[data-pid]').forEach(tr => {
            const pid = String(tr.dataset.pid);

            // als product niet (meer) gematcht is, alert weghalen
            const next = tr.nextElementSibling;
            if (!idSet.has(pid)) {
                if (next && next.matches('.product-alert-row') && next.dataset.pid === pid) next.remove();
                return;
            }

            const prod = productById(pid);
            const check = computePropertyCheck(prod, presentMap);
            renderProductAlertRow(pid, check);

            if (check.mismatches.length) {
                renderProductAlertRow(pid, check);
                // voeg toe:
                sendRejectIfNew(pid, prod, presentMap);
            } else {
                renderProductAlertRow(pid, check); // groen OK, geen foto
            }
        });
    }

    async function sendRejectIfNew(pid, prod, pmap) {
        const mism = detailedMismatches(prod, pmap);
        if (!mism.length) return;               // niets fout → niets sturen
        const key = JSON.stringify({ mism, pmap }); // eenvoudige dedupe
        if (lastRejectKeyByPid.get(pid) === key) return;
        lastRejectKeyByPid.set(pid, key);

        try {
            await POSTJSON('/api/rejects', {
                product_id: String(pid),
                product_name: String(prod?.name || pid),
                mismatches: mism,
                expected_properties: prod?.properties || {},
                present_counts: pmap,                // raw snapshot van wat er gezien werd
            });
        } catch (e) {
            console.warn('Kon reject niet opslaan:', e);
        }
    }

    // ====== helpers voor product-mismatch alerts ======
    function productById(pid) {
        return (products || []).find(p => String(p.id) === String(pid)) || null;
    }

    /**
     * Genereer foutmeldingen voor één product o.b.v. presentMap en product.properties.
     * - properties met waarde 0: overslaan (geen eis)
     * - bij != expected: foutregel toevoegen
     * Resultaat: [] of array met strings (NL melding)
     */
    function computePropertyMismatches(prod, pmap) {
        if (!prod || !prod.properties || typeof prod.properties !== 'object') return [];
        const msgs = [];
        for (const [name, expectedRaw] of Object.entries(prod.properties)) {
            const expected = Number(expectedRaw);
            if (!Number.isFinite(expected) || expected === 0) continue; // 0 = geen eis

            const actual = getActualCountFor(name, pmap);  // << tolerant tellen
            if (actual !== expected) {
                msgs.push(`Productiefout: ziet ${actual} ${name}, moet ${expected} ${name} zien.`);
            }
        }
        return msgs;
    }



    // ===== WebSocket verbinding met backend =====
    // Protocol kiezen (wss op https, ws anders)
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);

    // Eenvoudige handshake/keep-alive: stuur bij open 1 berichtje
    ws.onopen = () => ws.send("hi");

    // Verwerk inkomende berichten van de server
    ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);

        // 1) 'config': server pusht huidige config + (optioneel) classes & active_models
        if (msg.type === "config") {
            currentConfig = msg.config || {};

            // Vul sliders/inputs met de actuele waarden
            if (confInp && currentConfig.conf != null) confInp.value = currentConfig.conf;
            if (iouInp && currentConfig.iou != null) iouInp.value = currentConfig.iou;
            if (imgszInp && currentConfig.imgsz != null) imgszInp.value = currentConfig.imgsz;
            if (holdInp && currentConfig.hold_ms != null) holdInp.value = currentConfig.hold_ms;
            if (hitsInp && currentConfig.min_hits != null) hitsInp.value = currentConfig.min_hits;
            if (emaInp && currentConfig.ema_alpha != null) emaInp.value = currentConfig.ema_alpha;
            if (showMasksInp && typeof msg?.config?.show_masks === "boolean") { showMasksInp.checked = !!msg.config.show_masks; }
            if (showBoxesInp && typeof msg?.config?.show_boxes === "boolean") {
                showBoxesInp.checked = !!msg.config.show_boxes;
            }
            if (showContoursInp && typeof msg?.config?.contour_match_enabled === "boolean") {
                showContoursInp.checked = !!msg.config.contour_match_enabled;
            }

            // (Re)genereer class-checkboxen op basis van 'classes' van het primary model
            if (classesBox && Array.isArray(msg.classes)) {
                // allowed_classes: leeg betekent "alles toegestaan"
                const allowed = new Set((currentConfig.allowed_classes || []).map(s => s.toLowerCase()));
                classesBox.innerHTML = "";
                msg.classes.forEach(label => {
                    const id = `cls_${label.replace(/\W+/g, '_')}`; // DOM-id veilig maken
                    const wrap = document.createElement('div');
                    wrap.className = 'form-check';
                    wrap.innerHTML = `
                      <input type="checkbox" class="form-check-input" id="${id}" value="${esc(label)}"
                             ${allowed.size === 0 ? 'checked' : (allowed.has(label.toLowerCase()) ? 'checked' : '')}>
                      <label class="form-check-label" for="${id}">${esc(label)}</label>
                    `;
                    classesBox.appendChild(wrap);
                });
            }
            return; // klaar met 'config' bericht
        }

        // 2) 'detections': live lijst met items + present counts (badge-overzicht)
        if (msg.type === "detections") {
            renderDetections(msg.items || []);
            renderPresent(msg.present || []);
            // Controleer of het beeld gecentreerd is en stabiel blijft
            const centered = isCentered(msg);

            if (Array.isArray(msg.matched_product_ids) && msg.matched_product_ids.length) {
                const pid = String(msg.matched_product_ids[0]);
                if (centered) {
                    if (autoPhotoMode && centered && elapsed >= STABILITY_MS) {
                        const now = Date.now();
                        if (now - lastAutoCapture > AUTO_CAPTURE_INTERVAL_MS) {
                            lastAutoCapture = now;
                            takePhoto();
                        }
                    }

                    if (lastCenteredProductId === pid) {
                        // al even gecentreerd → check timer
                        if (!stableCenterStart) stableCenterStart = Date.now();
                        const elapsed = Date.now() - stableCenterStart;
                        if (elapsed >= STABILITY_MS) {
                            // stabiel en gecentreerd → pas echt highlighten & keuren
                            highlightProducts(msg.matched_product_ids);
                            updateAllProductAlerts(msg.matched_product_ids);
                            renderTopMatchedProduct(msg.matched_product_ids);
                        } else {
                            renderTopMatchedProduct([], 'centreren…');
                        }
                    } else {
                        // nieuw product → reset timer
                        lastCenteredProductId = pid;
                        stableCenterStart = Date.now();
                        renderTopMatchedProduct([], 'centreren…');
                    }
                } else {
                    // niet gecentreerd → reset
                    stableCenterStart = null;
                    renderTopMatchedProduct([], 'plaats product in midden…');
                }
            } else {
                // geen match
                stableCenterStart = null;
                renderTopMatchedProduct([], '');
            }


            drawPOIOverlay(
                flattenMarkers(msg.poi_markers),
                {
                    frameW: msg.frame_w ?? msg.frameW ?? msg.w,
                    frameH: msg.frame_h ?? msg.frameH ?? msg.h
                }
            );

            if (Array.isArray(msg.contours) && msg.contours.length) {
                const box = document.getElementById("stickyBox");
                const div = document.createElement('div');
                div.className = 'mt-2';
                div.innerHTML = '<div class="small text-muted">Contour matches:</div>' +
                    msg.contours.map(h => `<span class="badge text-bg-success me-1">${h.type_key} ${h.iop}</span>`).join(' ');
                box.appendChild(div);
            }
            return;
        }

    };

    function highlightProducts(ids) {
        // Normaliseer wat er binnenkomt
        let arr = [];
        if (Array.isArray(ids)) arr = ids;
        else if (ids instanceof Set) arr = Array.from(ids);
        else if (typeof ids === 'string' || typeof ids === 'number') arr = [String(ids)];
        else if (ids && typeof ids === 'object') {
            if (Array.isArray(ids.matched_product_ids)) arr = ids.matched_product_ids;
            else if (Array.isArray(ids.ids)) arr = ids.ids;
        }

        arr = arr.map(String);
        lastMatchedProductIds = new Set(arr);

        // pas kleur toe
        document.querySelectorAll('#productsBody tr.product-row[data-pid]').forEach(tr => {
            const pid = String(tr.dataset.pid);
            if (lastMatchedProductIds.has(pid)) {
                tr.classList.add('product-hit');
            } else {
                tr.classList.remove('product-hit');
            }
        });
    }




    // ===== Render-functies =====

    /**
     * Render de tabel met individuele detecties (label + confidence%).
     * Verwacht items: [{label: string, conf: number}, ...] met conf in %
     */
    function renderDetections(items) {
        if (!tbody) return;
        tbody.innerHTML = "";

        if (!items.length) {
            // Lege staat voor betere UX
            tbody.innerHTML = `<tr><td colspan="2" class="text-muted">No objects</td></tr>`;
            return;
        }
        for (const it of items) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${esc(it.label)}</td><td>${Number(it.conf).toFixed(1)}%</td>`;
            tbody.appendChild(tr);
        }
    }

    /**
     * Render het "present" badge-overzicht (sticky rechtsboven).
     * Verwacht arr: [{label: string, count: number}, ...]
     */
    function renderPresent(arr) {
        presentMap = {};
        // sla genormaliseerd op
        arr.forEach(s => presentMap[norm(s.label)] = Number(s.count) || 0);

        if (stickyBox) {
            if (!arr.length) {
                stickyBox.textContent = "Geen objecten in beeld…";
            } else {
                stickyBox.innerHTML = "";
                for (const s of arr) {
                    const span = document.createElement("span");
                    span.className = "badge text-bg-secondary me-1";
                    span.textContent = `${s.label} × ${s.count}`;
                    stickyBox.appendChild(span);
                }
            }
        }
    }


    // ===== Opslaan van detector-config (conf/iou/imgsz/allowed_classes) =====
    async function saveConfig() {
        // allowed_classes: als ALLE checkboxes aan staan -> stuur lege lijst (betekent "geen filter")
        const allowed = [];
        if (classesBox) {
            classesBox.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                if (cb.checked) allowed.push(cb.value);
            });
        }
        const total = classesBox ? classesBox.querySelectorAll('input[type="checkbox"]').length : 0;
        const allowed_classes = (total && allowed.length === total) ? [] : allowed;

        // Backend verwacht alleen velden die je wilt aanpassen
        await POSTJSON("/api/config", {
            conf: confInp ? parseFloat(confInp.value) : undefined,
            iou: iouInp ? parseFloat(iouInp.value) : undefined,
            imgsz: imgszInp ? parseInt(imgszInp.value, 10) : undefined,
            allowed_classes,
            show_masks: showMasksInp ? !!showMasksInp.checked : undefined
        });
    }

    // ===== Opslaan van stabilisatie-parameters (hold_ms/min_hits/ema_alpha) =====
    async function saveStab() {
        await POSTJSON("/api/config", {
            hold_ms: holdInp ? parseInt(holdInp.value, 10) : undefined,
            min_hits: hitsInp ? parseInt(hitsInp.value, 10) : undefined,
            ema_alpha: emaInp ? parseFloat(emaInp.value) : undefined,
        });
    }

    // ===== Collections & Products (read-only in viewer) =====

    /**
     * Haal collections op en render de lijst (links). Activeer filtering op click.
     */
    async function loadCollections() {
        if (!collectionsList) return;
        try {
            collections = await GET('/collections');
            renderCollections();
        } catch (e) {
            collectionsList.innerHTML = `<div class="list-group-item text-danger">Fout bij laden</div>`;
        }
    }

    /**
     * Haal producten op (eventueel gefilterd op collection_id) en render de tabel.
     */
    async function loadProducts(collection_id = null) {
        if (!productsBody) return;
        try {
            const url = collection_id ? `/products?collection_id=${encodeURIComponent(collection_id)}` : '/products';
            products = await GET(url);
            renderProducts();
        } catch (e) {
            productsBody.innerHTML = `<tr><td colspan="2" class="text-danger">Fout bij laden</td></tr>`;
        }
    }

    /**
     * Render de lijst met collections als selecteerbare knoppen (Bootstrap list-group).
     */
    function renderCollections() {
        if (!collectionsList) return;
        collectionsList.innerHTML = '';
        if (!collections.length) {
            collectionsList.innerHTML = `<div class="list-group-item text-muted">Geen collections</div>`;
            return;
        }

        collections.forEach(c => {
            const btn = document.createElement('button');
            btn.type = 'button';
            // 'active' klasse markeert de huidige selectie
            btn.className = `list-group-item list-group-item-action ${selectedCollectionId === c.id ? 'active' : ''}`;
            btn.textContent = c.name;
            btn.addEventListener('click', async () => {
                selectedCollectionId = c.id;  // update selectie
                renderCollections();          // herteken lijst (zorgt voor juiste 'active' styling)
                await loadProducts(c.id);     // en laad gefilterde producten
            });
            collectionsList.appendChild(btn);
        });
    }

    /**
     * Render de producttabel. Properties worden als 'chips' (badges) getoond.
     */
    function renderProducts() {
        if (!productsBody) return;
        productsBody.innerHTML = '';

        if (!products || !products.length) {
            productsBody.innerHTML = `<tr><td colspan="2" class="text-muted">Geen producten</td></tr>`;
        } else {
            for (const p of products) {
                const props = p?.properties && typeof p.properties === 'object'
                    ? Object.entries(p.properties)
                        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
                        .map(([k, v]) => `<span class="badge text-bg-light me-1">${esc(k)}: ${esc(String(v))}</span>`)
                        .join(' ')
                    : '';

                const tr = document.createElement('tr');
                tr.classList.add('product-row', 'align-middle');
                tr.dataset.pid = String(p.id);          // << needed for highlight
                tr.innerHTML = `
        <td>${esc(p.name || '')}</td>
        <td>${props || '—'}</td>
      `;
                productsBody.appendChild(tr);
            }
        }

        if (productsCount) productsCount.textContent = String(products?.length || 0);

        // keep highlights after re-render
        highlightProducts(Array.from(lastMatchedProductIds));
        updateAllProductAlerts(Array.from(lastMatchedProductIds));
    }

    // ===== Model-lijsten (basismodellen + getrainde) & acties =====

    /**
     * Veilig ophalen van actieve modellen; ondersteunt oudere backends die 404 geven.
     */
    async function getActiveModelsSafe() {
        try {
            const r = await fetch('/api/models/active');
            if (!r.ok) {
                if (r.status === 404) return { active: [] }; // compat met oudere backend zonder deze route
                throw new Error(`HTTP ${r.status}`);
            }
            return await r.json();
        } catch {
            return { active: [] };
        }
    }

    /**
     * Laad:
     *  - /api/config (lijst basismodellen + huidige model_key)
     *  - /api/models (getrainde modellen)
     *  - /api/models/active (actieve volgorde: primary = eerste)
     * En render de UI (radio/checkboxen + badges).
     */
    async function loadModelLists() {
        try {
            const cfg = await GET('/api/config');
            const baseModels = Array.isArray(cfg?.models) ? cfg.models : [];
            const activeKey = cfg?.config?.model_key || null;

            // --- Basismodellen (radio) ---
            if (baseModelsBox) {
                baseModelsBox.innerHTML = '';
                if (!baseModels.length) {
                    baseModelsBox.innerHTML = `<div class="text-muted small">Geen basismodellen gevonden.</div>`;
                } else {
                    baseModels.forEach(key => {
                        const id = `base_${key}`;
                        const div = document.createElement('div');
                        div.className = 'form-check';
                        div.innerHTML = `
                          <input class="form-check-input" type="radio" name="baseModelRadio" id="${id}" value="${esc(key)}"
                                 ${key === activeKey ? 'checked' : ''}>
                          <label class="form-check-label" for="${id}">
                            ${esc(key)} ${key === activeKey ? '<span class="badge text-bg-success ms-1">actief</span>' : ''}
                          </label>
                        `;
                        baseModelsBox.appendChild(div);
                    });
                }
            }

            // --- Getrainde modellen (checkbox) + badges 'primary' of 'actief' ---
            const trained = await GET('/api/models');        // lijst van {name, path, size, mtime, source}
            const activeRes = await getActiveModelsSafe();      // { active: [paths/keys...] } (eerste = primary)
            const activeArr = Array.isArray(activeRes?.active) ? activeRes.active.map(String) : [];
            const activeSet = new Set(activeArr);
            const primary = activeArr[0] || null;

            if (trainedModelsBox) {
                trainedModelsBox.innerHTML = '';
                if (!Array.isArray(trained) || !trained.length) {
                    trainedModelsBox.innerHTML = `<div class="text-muted small">Nog geen getrainde modellen.</div>`;
                } else {
                    trained.forEach((m, idx) => {
                        const id = `trained_${idx}`;
                        const sizeKB = Math.round((m.size || 0) / 1024);
                        const pathStr = String(m.path);
                        const isActive = activeSet.has(pathStr);
                        const isPrimary = primary === pathStr;

                        const div = document.createElement('div');
                        div.className = 'form-check';
                        div.innerHTML = `
                          <input class="form-check-input" type="checkbox"
                                 name="trainedModelCheck" id="${id}" value="${esc(pathStr)}"
                                 ${isActive ? 'checked' : ''}>
                          <label class="form-check-label" for="${id}">
                            ${esc(m.name)}
                            <small class="text-muted">(${sizeKB} KB)</small>
                            ${isPrimary ? '<span class="badge text-bg-success ms-1">primary</span>'
                                : (isActive ? '<span class="badge text-bg-info ms-1">actief</span>' : '')}
                          </label>
                        `;
                        trainedModelsBox.appendChild(div);
                    });
                }
            }
        } catch (e) {
            // Toon nette foutmeldingen in de UI-secties
            console.error('Kon modellen niet laden:', e);
            if (baseModelsBox) baseModelsBox.innerHTML = `<div class="text-danger small">Fout bij laden basismodellen</div>`;
            if (trainedModelsBox) trainedModelsBox.innerHTML = `<div class="text-danger small">Fout bij laden getrainde modellen</div>`;
        }
    }

    // --- Actie: basismodel toepassen (radio selectie -> POST /api/config model_key)
    applyBaseModelBtn?.addEventListener('click', async () => {
        const sel = baseModelsBox?.querySelector('input[name="baseModelRadio"]:checked')?.value;
        if (!sel) return; // niets geselecteerd
        try {
            await POSTJSON('/api/config', { model_key: sel });
            await loadModelLists(); // UI refresh (badges/checked)
        } catch (e) {
            alert('Kon basismodel instellen: ' + e.message);
        }
    });

    // --- Actie: getrainde modellen toepassen (multi-select) ---
    applyTrainedModelBtn?.addEventListener('click', async () => {
        if (!trainedModelsBox) return;
        // Verzamel alle aangevinkte modellen (paths)
        const checks = Array.from(trainedModelsBox.querySelectorAll('input[name="trainedModelCheck"]:checked') || []);
        if (!checks.length) return alert('Kies minstens één getraind model.');
        const selected = checks.map(el => el.value);

        // Nieuwe (multi-model) backend: POST /api/models/active
        const resp = await fetch('/api/models/active', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ models: selected })
        });

        if (resp.status === 404) {
            // Oudere backend (alleen single-model): fallback naar /api/model/load met eerste selectie
            const first = selected[0];
            try {
                const res = await POSTJSON('/api/model/load', { path: first });
                if (res?.error) alert('Kon model niet laden: ' + res.error);
            } catch (e) {
                alert('Kon model niet laden: ' + e.message);
            }
        } else {
            const data = await resp.json().catch(() => ({}));
            if (data?.error) alert('Kon actieve modellen niet instellen: ' + data.error);
        }

        try { await loadModelLists(); } catch { /* stil */ }
    });

    // ===== Event bindings =====
    if (saveBtn) saveBtn.addEventListener("click", saveConfig);
    if (saveStabBtn) saveStabBtn.addEventListener("click", saveStab);

    // Snelkeuzes voor class-filters
    allowAllBtn?.addEventListener("click", () =>
        classesBox?.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true)
    );
    clearAllBtn?.addEventListener("click", () =>
        classesBox?.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false)
    );

    // ===== Init-sequentie =====
    // Laad bij start:
    //  - model-lijsten (basismodellen + getrainde + active/primary badges)
    //  - collections & products (read-only)
    (async () => {
        await loadModelLists();
        await loadCollections();
        await loadProducts(null);
        await loadContoursList();
        highlightProducts(lastMatchedProductIds);
    })();
})();