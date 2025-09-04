// ===============================
// viewer.js
// ===============================
// Taken van dit script:
// - Opzetten van de WebSocket verbinding met de server (/ws)
// - Ontvangen van configuratie en detectieresultaten
// - UI updaten: tabel met detecties, "In beeld" teller, sliders, checkboxes, help-modal
// - Versturen van nieuwe configuratie naar de server (/api/config)
//
// Let op: sommige UI-elementen (sliders, checkboxes, knoppen) zijn optioneel,
// afhankelijk van of ze in viewer.html gedefinieerd zijn.
// ===============================

(function () {
  // Element-referenties uit de DOM (HTML-pagina)
  const tbody = document.getElementById("detBody");      // tabelbody waar detecties getoond worden
  const stickyBox = document.getElementById("stickyBox"); // box waar we "In beeld" aantallen tonen

  // Optionele UI-elementen voor modelkeuze en sliders
  const modelSel  = document.getElementById("modelSel");
  const confInp   = document.getElementById("confInp");
  const confVal   = document.getElementById("confVal");
  const iouInp    = document.getElementById("iouInp");
  const iouVal    = document.getElementById("iouVal");
  const imgszInp  = document.getElementById("imgszInp");
  const imgszVal  = document.getElementById("imgszVal");
  const classesBox = document.getElementById("classesBox");
  const saveBtn   = document.getElementById("saveBtn");
  const allowAllBtn = document.getElementById("allowAllBtn");
  const clearAllBtn = document.getElementById("clearAllBtn");

  // Modal (helpvenster met uitleg)
  const helpBtn = document.getElementById("helpBtn");
  const helpModal = document.getElementById("helpModal");
  const closeModal = document.getElementById("closeModal");

  // --- Event handlers voor help-modal ---
  if (helpBtn) helpBtn.onclick = () => { helpModal.style.display = "block"; };
  if (closeModal) closeModal.onclick = () => { helpModal.style.display = "none"; };
  window.onclick = (event) => {
    if (event.target === helpModal) {
      helpModal.style.display = "none";
    }
  };

  // --- Kleine helperfunctie om sliders live waarde te laten tonen ---
  const bindRange = (inp, label, fmt = (v)=>v) => {
    if (!inp || !label) return;
    const update = () => label.textContent = fmt(inp.value);
    inp.addEventListener("input", update);
    update();
  };
  // Koppel sliders aan labels
  bindRange(confInp, confVal, v => (+v).toFixed(2));
  bindRange(iouInp,  iouVal,  v => (+v).toFixed(2));
  bindRange(imgszInp,imgszVal, v => `${v}`);

  // --- WebSocket verbinding met server ---
  // Gebruikt wss:// bij https, anders ws://
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => ws.send("hi"); // simpele handshake/ping naar server

  // Verwerken van berichten die binnenkomen via de WS
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);

    // Init/config bericht (verstuurd bij connect of na config-update)
    if (msg.type === "config") {
      const cfg = msg.config || {};
      const models = msg.models || [];
      const classes = msg.classes || null;

      // Model dropdown vullen
      if (modelSel) {
        modelSel.innerHTML = "";
        for (const m of models) {
          const opt = document.createElement("option");
          opt.value = m; opt.textContent = m;
          modelSel.appendChild(opt);
        }
        if (cfg.model_key) modelSel.value = cfg.model_key;
      }

      // Sliders initialiseren met huidige waarden
      if (confInp && confVal) { confInp.value = cfg.conf ?? 0.35; confVal.textContent = (+confInp.value).toFixed(2); }
      if (iouInp && iouVal)   { iouInp.value  = cfg.iou ?? 0.45;  iouVal.textContent  = (+iouInp.value).toFixed(2); }
      if (imgszInp && imgszVal){ imgszInp.value = cfg.imgsz ?? 640; imgszVal.textContent = `${imgszInp.value}`; }

      // Klassen checkboxes invullen
      if (classesBox && classes && Array.isArray(classes)) {
        renderClassCheckboxes(classes, cfg.allowed_classes || []);
      }
      return; // klaar met configbericht
    }

    // Detectie-bericht (komt elke frame binnen met huidige detecties)
    if (msg.type === "detections") {
      renderDetections(msg.items || []);  // tabel met individuele detecties
      renderPresent(msg.present || []);   // "In beeld" aantallen (stabiel)
      return;
    }
  };

  // --- UI-render functies ---

  // Vul de detectietabel met rijen: label + confidence
  function renderDetections(items) {
    tbody.innerHTML = "";
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="2">No objects</td></tr>`;
      return;
    }
    for (const it of items) {
      const tr = document.createElement("tr");
      const tdL = document.createElement("td");
      const tdC = document.createElement("td");
      tdL.textContent = it.label;
      tdC.textContent = `${it.conf}%`;
      tr.appendChild(tdL);
      tr.appendChild(tdC);
      tbody.appendChild(tr);
    }
  }

  // Toon stabiele tellingen ("present"): bijv. "tv × 2", "person × 3"
  function renderPresent(arr) {
    if (!stickyBox) return;
    if (!arr || !arr.length) {
      stickyBox.textContent = "Geen objecten in beeld…";
      return;
    }
    stickyBox.innerHTML = "";
    for (const s of arr) {
      const span = document.createElement("span");
      span.className = "pill"; // CSS-klasse voor mooie badge
      span.textContent = `${s.label} × ${s.count}`;
      stickyBox.appendChild(span);
    }
  }

  // (Optioneel) render checkboxes voor alle klassen
  function renderClassCheckboxes(allClasses, allowed) {
    if (!classesBox) return;
    const allowedSet = new Set((allowed || []).map(x => x.toLowerCase()));
    classesBox.innerHTML = "";
    allClasses.forEach(label => {
      const id = `cls_${label.replace(/\W+/g, "_")}`;
      const wrap = document.createElement("label");
      wrap.className = "class-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = label;
      cb.id = id;
      // Als allowed leeg is → alles toegestaan (dus alle vakjes aangevinkt)
      cb.checked = (allowedSet.size === 0) ? true : allowedSet.has(label.toLowerCase());
      const txt = document.createElement("span");
      txt.textContent = label;
      wrap.appendChild(cb);
      wrap.appendChild(txt);
      classesBox.appendChild(wrap);
    });
  }

  // Config opslaan: stuur huidige UI-waarden naar /api/config
  async function saveConfig() {
    const allowed = [];
    if (classesBox) {
      classesBox.querySelectorAll('input[type="checkbox"]').forEach(cb => { if (cb.checked) allowed.push(cb.value); });
    }
    const total = classesBox ? classesBox.querySelectorAll('input[type="checkbox"]').length : 0;
    const allowed_classes = (total && allowed.length === total) ? [] : allowed; // leeg = alles

    const body = {
      model_key: modelSel ? modelSel.value : undefined,
      conf: confInp ? parseFloat(confInp.value) : undefined,
      iou: iouInp ? parseFloat(iouInp.value) : undefined,
      imgsz: imgszInp ? parseInt(imgszInp.value, 10) : undefined,
      allowed_classes
    };

    await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  }

  // Event listeners voor knoppen
  if (saveBtn) saveBtn.addEventListener("click", saveConfig);
  if (allowAllBtn) allowAllBtn.addEventListener("click", () => {
    classesBox?.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
  });
  if (clearAllBtn) clearAllBtn.addEventListener("click", () => {
    classesBox?.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
  });
})();
