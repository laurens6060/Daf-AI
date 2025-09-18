(function () {
  // ==========================
  // DOM referenties (elementen uit de pagina)
  // ==========================
  const tbody = document.getElementById("detBody");      // Tabel met individuele detecties (label + confidence)
  const stickyBox = document.getElementById("stickyBox"); // Badge-overzicht: “in beeld” (huidige aantallen per label)

  // Besturingselementen (sliders, dropdowns, etc.) — sommige pagina’s hebben niet alles
  const modelSel  = document.getElementById("modelSel");
  const confInp   = document.getElementById("confInp");   const confVal  = document.getElementById("confVal");
  const iouInp    = document.getElementById("iouInp");    const iouVal   = document.getElementById("iouVal");
  const imgszInp  = document.getElementById("imgszInp");  const imgszVal = document.getElementById("imgszVal");
  const holdInp   = document.getElementById("holdInp");   const holdVal  = document.getElementById("holdVal");
  const hitsInp   = document.getElementById("hitsInp");   const hitsVal  = document.getElementById("hitsVal");
  const emaInp    = document.getElementById("emaInp");    const emaVal   = document.getElementById("emaVal");
  const classesBox = document.getElementById("classesBox");
  const saveBtn    = document.getElementById("saveBtn");
  const allowAllBtn = document.getElementById("allowAllBtn");
  const clearAllBtn = document.getElementById("clearAllBtn");
  const saveStabBtn = document.getElementById("saveStab");

  // ID van de “actieve” order die we willen kleuren (blauw/rood/groen)
  const ACTIVE_ORDER_ID = "156547";

  // Orders-tabel (rechter kolom)
  const ordersBody = document.getElementById("ordersBody");

  // ==========================
  // Interne staat
  // ==========================
  let currentConfig = null;     // Laatst ontvangen config van de server (model, conf, iou, …)
  let orders = [];              // Alle orders uit /mes/orders
  let presentMap = {};          // “In beeld” aantallen per label, bv. { tv: 2, person: 1 }
  const axleCache = new Map();  // Cache per as-type: typeCode -> { spec, subtypesByName }

  // ==========================
  // Kleine helpers
  // ==========================
  // Verbind slider met zichtbaar label (zodat waarde live mee verandert)
  const bindRange = (inp, label, fmt = (v)=>v) => {
    if (!inp || !label) return;
    const update = () => label.textContent = fmt(inp.value);
    inp.addEventListener("input", update);
    update();
  };
  bindRange(confInp,  confVal,  v => (+v).toFixed(2));
  bindRange(iouInp,   iouVal,   v => (+v).toFixed(2));
  bindRange(imgszInp, imgszVal, v => `${v}`);
  bindRange(holdInp,  holdVal,  v => `${v} ms`);
  bindRange(hitsInp,  hitsVal,  v => `${v}`);
  bindRange(emaInp,   emaVal,   v => (+v).toFixed(2));

  // HTTP hulpfuncties
  async function GET(url){ return fetch(url).then(r=>r.json()); }
  async function POST(url, body){
    return fetch(url,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)
    }).then(r=>r.json());
  }

  // ==========================
  // Orders ophalen en tekenen
  // ==========================
  async function loadOrders() {
    if (!ordersBody) return; // als de kolom niet bestaat, sla over
    orders = await GET('/mes/orders');

    // Tabel leeg en opnieuw opbouwen
    ordersBody.innerHTML = "";
    for (const o of orders) {
      const tr = document.createElement('tr');
      tr.dataset.orderId = o.order_id; // nodig om later specifieke rij te kunnen kleuren
      tr.innerHTML = `<td>${o.order_id}</td><td>${o.axle_type}</td><td>${o.axle_subtype}</td>`;
      ordersBody.appendChild(tr);
    }

    // Voor alle type-codes in deze orders alvast de subtype-info cachen
    await Promise.all([...new Set(orders.map(o => o.axle_type))].map(cacheAxleType));

    // Eerste kleur-update (meestal “actief maar niks gezien” → blauw)
    updateOrdersHighlight();
  }

  // Cache hulp: as-type info (properties + subtypes) één keer ophalen en bewaren
  async function cacheAxleType(typeCode) {
    if (axleCache.has(typeCode)) return;
    const data = await GET(`/mes/axle-types/${encodeURIComponent(typeCode)}`);
    const byName = new Map();
    (data.subtypes || []).forEach(st => byName.set(String(st.name), st));
    axleCache.set(typeCode, { spec: data.spec, subtypesByName: byName });
  }

  // ==========================
  // Rijkleuren van de orders
  // ==========================
  // Regels:
  // - Actieve order: blauw als er nog niets van de vereiste properties gezien is
  // - Actieve order: groen als ALLE vereiste properties exact matchen (bv. tv == 2)
  // - Actieve order: rood als we wel iets van de vereiste properties zien, maar niet het doel-aantal
  // - Niet-actieve orders: grijs
  function updateOrdersHighlight() {
    if (!ordersBody || !orders.length) return;

    // 0) Alles eerst “grijs” (neutrale kleur voor niet-actieve orders)
    ordersBody.querySelectorAll('tr').forEach(tr => {
      tr.classList.remove('table-success', 'table-danger', 'table-primary', 'table-Light');
      tr.classList.add('table-Light'); // custom licht/grijs (of vervang door 'table-secondary' als je wil)
    });

    // 1) Zoek de rij van de actieve order
    const active = orders.find(o => String(o.order_id) === ACTIVE_ORDER_ID);
    if (!active) return;

    const tr = ordersBody.querySelector(`tr[data-order-id="${ACTIVE_ORDER_ID}"]`);
    if (!tr) return;

    // 2) Haal de vereisten van het subtype uit de cache (bv. { tv: 2 } of { person:1, tv:2 })
    const cached = axleCache.get(active.axle_type);
    const sub = cached?.subtypesByName?.get(String(active.axle_subtype));
    const req = (sub && sub.values) ? sub.values : {};

    const keys = Object.keys(req || {});
    if (keys.length === 0) {
      // Geen eisen voor dit subtype → laat grijs
      tr.classList.remove('table-success', 'table-danger', 'table-primary');
      tr.classList.add('table-Light');
      return;
    }

    // 3) Vergelijk “in beeld” (presentMap) met “vereist” (req)
    const allZeroDetected = keys.every(k => (presentMap[k] || 0) === 0); // niks gezien voor alle vereiste keys

    const allMatch =
      keys.length > 0 &&
      keys.every(k => {
        const need = Number(req[k]) || 0;
        const have = Number(presentMap[k] || 0);
        return need > 0 && have === need;
      });

    const anySeenWrong =
      keys.some(k => {
        const need = Number(req[k]) || 0;
        const have = Number(presentMap[k] || 0);
        return have > 0 && have !== need; // wél gezien, maar niet het doel
      });

    // 4) Pas kleur toe op de actieve rij
    tr.classList.remove('table-success', 'table-danger', 'table-primary', 'table-Light');

    if (allMatch) {
      tr.classList.add('table-success');   // ✅ exact goed
    } else if (allZeroDetected) {
      tr.classList.add('table-primary');   // 🔵 actief, maar nog niets gezien
    } else if (anySeenWrong) {
      tr.classList.add('table-danger');    // 🔴 we zien de property, maar niet het juiste aantal
    } else {
      tr.classList.add('table-Light');     // ◻️ overige gevallen → neutraal
    }
  }

  // ==========================
  // WebSocket: live updates van de server
  // ==========================
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => ws.send("hi"); // eenvoudige keepalive

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);

    // Init/config—bericht: vul UI-elementen en waardes
    if (msg.type === "config") {
      currentConfig = msg.config || {};

      // Modelkeuze
      if (modelSel) {
        modelSel.innerHTML = "";
        (msg.models || []).forEach(m => {
          const opt = document.createElement('option');
          opt.value = m; opt.textContent = m;
          modelSel.appendChild(opt);
        });
        if (currentConfig.model_key) modelSel.value = currentConfig.model_key;
      }

      // Sliders updaten met huidige waarden
      if (confInp && currentConfig.conf != null) confInp.value = currentConfig.conf;
      if (iouInp  && currentConfig.iou  != null) iouInp.value  = currentConfig.iou;
      if (imgszInp&& currentConfig.imgsz!= null) imgszInp.value= currentConfig.imgsz;
      if (holdInp && currentConfig.hold_ms   != null) holdInp.value = currentConfig.hold_ms;
      if (hitsInp && currentConfig.min_hits  != null) hitsInp.value = currentConfig.min_hits;
      if (emaInp  && currentConfig.ema_alpha != null) emaInp.value = currentConfig.ema_alpha;

      // Klassen-checkboxen opbouwen (optioneel)
      if (classesBox && Array.isArray(msg.classes)) {
        const allowed = new Set((currentConfig.allowed_classes || []).map(s => s.toLowerCase()));
        classesBox.innerHTML = "";
        msg.classes.forEach(label => {
          const id = `cls_${label.replace(/\W+/g,'_')}`;
          const wrap = document.createElement('div');
          wrap.className = 'form-check';

          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.className = 'form-check-input';
          cb.id = id;
          cb.value = label;
          cb.checked = allowed.size === 0 ? true : allowed.has(label.toLowerCase());

          const lab = document.createElement('label');
          lab.className = 'form-check-label';
          lab.setAttribute('for', id);
          lab.textContent = label;

          wrap.appendChild(cb);
          wrap.appendChild(lab);
          classesBox.appendChild(wrap);
        });
      }
      return;
    }

    // Realtime detecties: tabel + “in beeld” en vervolgens orders kleuren
    if (msg.type === "detections") {
      renderDetections(msg.items || []);
      renderPresent(msg.present || []);
      return;
    }
  };

  // ==========================
  // Rendering helpers
  // ==========================
  // Individuele detecties (onderaan: label + confidence)
  function renderDetections(items) {
    if (!tbody) return;
    tbody.innerHTML = "";
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="2">No objects</td></tr>`;
      return;
    }
    for (const it of items) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${it.label}</td><td>${it.conf}%</td>`;
      tbody.appendChild(tr);
    }
  }

  // “In beeld” badges + update van orderkleur op basis van huidige tellingen
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

    // Na elke update de actieve order opnieuw evalueren/inkleuren
    updateOrdersHighlight();
  }

  // ==========================
  // Opslaan van instellingen (POST /api/config)
  // ==========================
  async function saveConfig() {
    // Allowed classes: als álle checkboxen aanstaan sturen we een lege lijst (betekent “alles”)
    const allowed = [];
    if (classesBox) {
      classesBox.querySelectorAll('input[type="checkbox"]').forEach(cb => { if (cb.checked) allowed.push(cb.value); });
    }
    const total = classesBox ? classesBox.querySelectorAll('input[type="checkbox"]').length : 0;
    const allowed_classes = (total && allowed.length === total) ? [] : allowed;

    const body = {
      model_key: modelSel ? modelSel.value : undefined,
      conf:   confInp ? parseFloat(confInp.value)   : undefined,
      iou:    iouInp  ? parseFloat(iouInp.value)    : undefined,
      imgsz:  imgszInp? parseInt(imgszInp.value,10) : undefined,
      allowed_classes
    };
    await POST("/api/config", body);
  }

  // Alleen stabilisatie-parameters opslaan
  async function saveStab() {
    const body = {
      hold_ms:   holdInp ? parseInt(holdInp.value,10)  : undefined,
      min_hits:  hitsInp ? parseInt(hitsInp.value,10)  : undefined,
      ema_alpha: emaInp ?  parseFloat(emaInp.value)    : undefined,
    };
    await POST("/api/config", body);
  }

  // Buttons koppelen (indien aanwezig op de pagina)
  if (saveBtn)     saveBtn.addEventListener("click", saveConfig);
  if (saveStabBtn) saveStabBtn.addEventListener("click", saveStab);
  if (allowAllBtn) allowAllBtn.addEventListener("click", () => {
    classesBox?.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
  });
  if (clearAllBtn) clearAllBtn.addEventListener("click", () => {
    classesBox?.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
  });

  // ==========================
  // Init (start)
  // ==========================
  (async () => {
    // Vul de orders-tabel en warm de subtype-cache op
    await loadOrders();
  })();
})();
