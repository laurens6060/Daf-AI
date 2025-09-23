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
    const emaInp  = document.getElementById("emaInp");  const emaVal  = document.getElementById("emaVal");

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

    // ===== In-memory staat =====
    let currentConfig = null; // laatste 'config' van de server (als bron van waarheid)
    let presentMap = {};      // snelle lookup voor label -> count (optioneel voor toekomstige UI-logica)

    let collections = [];           // laatste geladen collections
    let products = [];              // laatste geladen producten (eventueel gefilterd op collection)
    let selectedCollectionId = null;

    // ===== Kleine helpers =====

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
    bindRange(confInp,  confVal,  v => (+v).toFixed(2));
    bindRange(iouInp,   iouVal,   v => (+v).toFixed(2));
    bindRange(imgszInp, imgszVal, v => `${v}`);
    bindRange(holdInp,  holdVal,  v => `${v} ms`);
    bindRange(hitsInp,  hitsVal,  v => `${v}`);
    bindRange(emaInp,   emaVal,   v => (+v).toFixed(2));

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
            if (confInp  && currentConfig.conf       != null) confInp.value  = currentConfig.conf;
            if (iouInp   && currentConfig.iou        != null) iouInp.value   = currentConfig.iou;
            if (imgszInp && currentConfig.imgsz      != null) imgszInp.value = currentConfig.imgsz;
            if (holdInp  && currentConfig.hold_ms    != null) holdInp.value  = currentConfig.hold_ms;
            if (hitsInp  && currentConfig.min_hits   != null) hitsInp.value  = currentConfig.min_hits;
            if (emaInp   && currentConfig.ema_alpha  != null) emaInp.value   = currentConfig.ema_alpha;

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
            return;
        }
    };

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
        arr.forEach(s => presentMap[s.label] = s.count);

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
            conf:  confInp  ? parseFloat(confInp.value)  : undefined,
            iou:   iouInp   ? parseFloat(iouInp.value)   : undefined,
            imgsz: imgszInp ? parseInt(imgszInp.value, 10) : undefined,
            allowed_classes
        });
    }

    // ===== Opslaan van stabilisatie-parameters (hold_ms/min_hits/ema_alpha) =====
    async function saveStab() {
        await POSTJSON("/api/config", {
            hold_ms:  holdInp ? parseInt(holdInp.value, 10)   : undefined,
            min_hits: hitsInp ? parseInt(hitsInp.value, 10)   : undefined,
            ema_alpha: emaInp ? parseFloat(emaInp.value)      : undefined,
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
                // Properties als badges: gesorteerd op sleutelnaam voor voorspelbare volgorde
                const props = p?.properties && typeof p.properties === 'object'
                    ? Object.entries(p.properties)
                        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
                        .map(([k, v]) => `<span class="badge text-bg-light me-1">${esc(k)}: ${esc(String(v))}</span>`)
                        .join(' ')
                    : '';

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${esc(p.name || '')}</td>
                    <td>${props || '—'}</td>
                `;
                productsBody.appendChild(tr);
            }
        }

        // Kleine teller boven de tabel
        if (productsCount) productsCount.textContent = String(products?.length || 0);
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
            const trained   = await GET('/api/models');        // lijst van {name, path, size, mtime, source}
            const activeRes = await getActiveModelsSafe();      // { active: [paths/keys...] } (eerste = primary)
            const activeArr = Array.isArray(activeRes?.active) ? activeRes.active.map(String) : [];
            const activeSet = new Set(activeArr);
            const primary   = activeArr[0] || null;

            if (trainedModelsBox) {
                trainedModelsBox.innerHTML = '';
                if (!Array.isArray(trained) || !trained.length) {
                    trainedModelsBox.innerHTML = `<div class="text-muted small">Nog geen getrainde modellen.</div>`;
                } else {
                    trained.forEach((m, idx) => {
                        const id = `trained_${idx}`;
                        const sizeKB = Math.round((m.size || 0) / 1024);
                        const pathStr = String(m.path);
                        const isActive  = activeSet.has(pathStr);
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
            if (baseModelsBox)    baseModelsBox.innerHTML    = `<div class="text-danger small">Fout bij laden basismodellen</div>`;
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
    if (saveBtn)     saveBtn.addEventListener("click", saveConfig);
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
        await loadProducts(null); // geen filter -> toon alle producten
    })();
})();
