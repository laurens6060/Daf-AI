const els = {
  pick: document.getElementById('pick'),
  file: document.getElementById('file'),
  work: document.getElementById('work'),
  img: document.getElementById('img'),
  cvs: document.getElementById('cvs'),
  undo: document.getElementById('undo'),
  clear: document.getElementById('clearPoly'),
  save: document.getElementById('save'),
  name: document.getElementById('name'),
  typeKey: document.getElementById('typeKey'),
  // lijst-sectie
  list: document.getElementById('list'),
  search: document.getElementById('search'),
  refresh: document.getElementById('refresh'),
  listCount: document.getElementById('listCount'),
  // config
  cls: document.getElementById('cls'),
  thr: document.getElementById('thr'),
  applyCfg: document.getElementById('applyCfg'),
  product: document.getElementById('product'),
};

let poly = [];
let W = 0, H = 0;
let _allContours = [];

let currentContourId = null;

const ctx = () => els.cvs.getContext('2d');

let _products = [];
let _prodById = new Map();

els.autoHole = document.getElementById('autoHole');
els.autoClass = document.getElementById('autoClass');

els.autoHole.onclick = async () => {
  try {
    const fd = new FormData();
    const cls = (els.autoClass.value || '').trim() || null;
    if (cls) fd.append('target_class', cls);

    // ← als er een foto gekozen is, meesturen:
    const picked = els.file?.files?.[0];
    if (picked) fd.append('file', picked, picked.name);

    const r = await fetch('/api/contours/auto', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data?.error || 'Automatische contour niet gevonden');

    // prefill canvas
    W = data.width; H = data.height;
    els.work.classList.remove('d-none');
    poly = data.polygon01.map(([x01, y01]) => [x01 * W, y01 * H]);
    draw();
  } catch (e) {
    console.error(e);
    alert(e.message);   // toont nu “Invalid image”, “No polygon found”, …
  }
};


// --- POI tool
els.poiLabel = document.getElementById('poiLabel');
els.poiReq = document.getElementById('poiReq');
els.poiR01 = document.getElementById('poiR01');
els.poiAddMode = document.getElementById('poiAddMode');
els.poiSave = document.getElementById('poiSave');
els.poiList = document.getElementById('poiList');

let poiAddActive = false;
let poiItems = []; // {x01,y01, expected_label, required, radius01}

function renderPOIList() {
  els.poiList.innerHTML = poiItems.map((p, i) =>
    `<li>#${i + 1} ${p.expected_label} @ (${p.x01.toFixed(3)},${p.y01.toFixed(3)}) r=${p.radius01.toFixed(3)} req=${p.required} 
      <button data-i="${i}" class="btn btn-link btn-sm text-danger">verwijder</button></li>`).join('');
  els.poiSave.disabled = !(poiItems.length && els.product.value);
  els.poiList.querySelectorAll('button[data-i]').forEach(b => {
    b.onclick = () => { poiItems.splice(parseInt(b.dataset.i, 10), 1); renderPOIList(); };
  });
}

els.poiAddMode.onclick = () => {
  poiAddActive = !poiAddActive;
  els.poiAddMode.classList.toggle('btn-outline-secondary', !poiAddActive);
  els.poiAddMode.classList.toggle('btn-secondary', poiAddActive);
  els.poiAddMode.textContent = poiAddActive ? 'POI plaatsen (actief)' : 'POI plaatsen';
  els.cvs.style.cursor = poiAddActive ? 'crosshair' : 'default';
};

// opslaan set
els.poiSave.onclick = async () => {
  const product_id = (els.product?.value || '') || null;
  if (!product_id) { alert('Kies eerst een product'); return; }
  const name = (els.name.value || '').trim() || 'POI-set';

  const body = {
    name,
    product_id,
    contour_id: currentContourId || null,   // ← koppel set aan de actieve contour
    items: poiItems
  };
  try {
    const r = await fetch('/api/pois', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!r.ok) throw new Error(await r.text());
    alert('POI-set opgeslagen');
    poiItems = []; renderPOIList();
  } catch (e) {
    console.error(e); alert('Opslaan mislukt');
  }
};


async function loadPOIsForContour(contour_id) {
  const r = await fetch('/api/pois?contour_id=' + encodeURIComponent(contour_id));
  const sets = await r.json();
  // Flatten alle items (je kan er later voor kiezen om per set te kiezen)
  poiItems = sets.flatMap(s => s.items || []);
  renderPOIList();
  draw();
}

function loadContourToEditor(c) {
  currentContourId = c.id;
  W = c.width; H = c.height;
  els.work.classList.remove('d-none');

  // referentie-afbeelding (indien opgeslagen)
  if (c.image_url) {
    els.img.onload = () => { draw(); };
    els.img.src = c.image_url;   // wordt door <img> getoond onder het canvas
  } else {
    els.img.removeAttribute('src');
  }

  // contour-poly op canvas zetten
  poly = (c.polygon01 || []).map(([x01, y01]) => [x01 * W, y01 * H]);

  // bijbehorende POI’s (optioneel gefilterd op dezelfde contour)
  loadPOIsForContour(c.id);
  draw();
}


async function loadProducts() {
  const r = await fetch('/products');
  _products = await r.json();
  _prodById = new Map(_products.map(p => [p.id, p]));
  els.product.innerHTML = '<option value="">(geen)</option>' +
    _products.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
}

/* ---------- lijst renderen + CRUD ---------- */
function renderList(items) {
  els.list.innerHTML = '';
  els.listCount.textContent = `${items.length} contour${items.length === 1 ? '' : 'en'}`;

  if (!items.length) {
    els.list.innerHTML = '<div class="text-muted">Nog geen contouren.</div>';
    return;
  }

  items.forEach(it => {
    const productName = it.product_id ? (_prodById.get(it.product_id)?.name || '(onbekend product)') : '(geen product)';

    const div = document.createElement('div');
    div.className = 'border rounded p-2 mb-2';
    div.innerHTML = `
  <div class="d-flex justify-content-between align-items-center">
    <div class="me-2">
      <div><strong>${it.name}</strong> <span class="text-muted">(${it.type_key})</span></div>
      <div class="small text-muted">${it.width}×${it.height} · ${productName}</div>
    </div>
    <div class="d-flex gap-2">
      <button class="btn btn-sm btn-outline-primary edit">Bewerken</button>
      <button class="btn btn-sm btn-outline-danger del">Verwijderen</button>
    </div>
  </div>
`;

    const btnEdit = div.querySelector('.edit');
    btnEdit.onclick = () => loadContourToEditor(it);

    const btnDel = div.querySelector('.del');
    btnDel.onclick = async () => {
      if (!confirm(`Contour “${it.name}” verwijderen? Dit kan niet ongedaan gemaakt worden.`)) return;
      btnDel.disabled = true; btnDel.textContent = 'Verwijderen…';
      try {
        const r = await fetch('/api/contours/' + it.id, { method: 'DELETE' });
        if (!r.ok) throw new Error(await r.text());
        _allContours = _allContours.filter(c => c.id !== it.id);
        applySearchAndRender();
      } catch (e) {
        console.error(e);
        alert('Verwijderen mislukt');
        btnDel.disabled = false; btnDel.textContent = 'Verwijderen';
      }
    };

    els.list.appendChild(div);
  });
}





function applySearchAndRender() {
  const q = (els.search.value || '').trim().toLowerCase();
  const filtered = q
    ? _allContours.filter(c =>
      (c.name || '').toLowerCase().includes(q) ||
      (c.type_key || '').toLowerCase().includes(q))
    : _allContours;
  renderList(filtered);
}

async function loadList() {
  els.list.innerHTML = '<div class="text-muted">Laden…</div>';
  try {
    const r = await fetch('/api/contours');
    if (!r.ok) throw new Error(await r.text());
    const items = await r.json();
    // sorteer: naam, dan type_key
    _allContours = items.slice().sort((a, b) => {
      const an = (a.name || '').toLowerCase(), bn = (b.name || '').toLowerCase();
      if (an !== bn) return an < bn ? -1 : 1;
      const at = (a.type_key || '').toLowerCase(), bt = (b.type_key || '').toLowerCase();
      return at < bt ? -1 : at > bt ? 1 : 0;
    });
    applySearchAndRender();
  } catch (e) {
    console.error(e);
    els.list.innerHTML = `<div class="text-danger">Laden mislukt</div>`;
    els.listCount.textContent = '';
  }
}

/* ---------- tekenen ---------- */
function drawPOIs(c) {
  if (!poiItems.length) return;
  c.save();

  // stijl
  const stroke = '#0dcaf0';          // bootstrap info-blauw
  const fill = 'rgba(13, 202, 240, 0.08)';

  poiItems.forEach(p => {
    const x = p.x01 * W;
    const y = p.y01 * H;
    const r = Math.max(4, Math.round(Math.min(W, H) * (p.radius01 || 0.02))); // px
    const s = Math.max(8, r * 2);  // boxzijde

    // box
    c.fillStyle = fill;
    c.strokeStyle = stroke;
    c.lineWidth = Math.max(1, Math.round(W / 600));

    c.beginPath();
    c.rect(Math.round(x - s / 2), Math.round(y - s / 2), s, s);
    c.fill();
    c.stroke();

    // centerpunt
    c.beginPath();
    c.arc(x, y, Math.max(2, Math.round(W / 300)), 0, Math.PI * 2);
    c.fillStyle = stroke;
    c.fill();

    // label (bv. "earbud × 1")
    const text = `${p.expected_label} × ${p.required}`;
    c.font = `bold ${Math.max(10, Math.round(W / 50))}px system-ui, sans-serif`;
    c.textBaseline = 'top';
    const pad = Math.max(2, Math.round(W / 300));
    const tx = Math.round(x + s / 2 + pad);
    const ty = Math.round(y - s / 2);

    // label achtergrond
    const metrics = c.measureText(text);
    const th = Math.round(parseInt(c.font, 10) * 1.3);
    const tw = Math.round(metrics.width + pad * 2);
    c.fillStyle = 'rgba(0,0,0,0.55)';
    c.fillRect(tx, ty, tw, th);

    // label tekst
    c.fillStyle = '#fff';
    c.fillText(text, tx + pad, ty + Math.round(pad / 2));
  });

  c.restore();
}

function draw() {
  const c = ctx();
  els.cvs.width = W; els.cvs.height = H;
  c.clearRect(0, 0, W, H);

  // poly (alleen schets tonen als er punten zijn)
  if (poly.length) {
    c.beginPath();
    c.moveTo(poly[0][0], poly[0][1]);
    for (let i = 1; i < poly.length; i++) c.lineTo(poly[i][0], poly[i][1]);
    c.strokeStyle = '#ffcc38';
    c.lineWidth = Math.max(2, Math.round(W / 400));
    c.setLineDash([10, 6]);
    c.stroke();
    c.setLineDash([]);
    c.fillStyle = '#ffcc38';
    for (const p of poly) { c.beginPath(); c.arc(p[0], p[1], Math.max(3, W / 300), 0, Math.PI * 2); c.fill(); }
  }

  // POI overlay (vierkante box + label)
  drawPOIs(c);

  // UI-knoppen en geldigheid
  els.undo.disabled = poly.length === 0;
  els.clear.disabled = poly.length === 0;
  els.save.disabled = !(poly.length >= 3 && els.name.value.trim() && els.typeKey.value.trim());
}


/* ---------- events tekenen ---------- */
els.pick.onclick = () => els.file.click();
els.file.onchange = () => {
  const f = els.file.files[0]; if (!f) return;
  const url = URL.createObjectURL(f);
  els.img.onload = () => {
    W = els.img.naturalWidth; H = els.img.naturalHeight;
    els.work.classList.remove('d-none');
    poly = []; draw();
  };
  els.img.src = url;
};
function toXY(ev) {
  const r = els.cvs.getBoundingClientRect();
  const x = (ev.clientX - r.left) * (W / r.width);
  const y = (ev.clientY - r.top) * (H / r.height);
  return [x, y];
}

// ➊ Polygoon: alleen wanneer NIET in POI-modus
els.cvs.addEventListener('mousedown', ev => {
  if (!W || !H) return;
  if (poiAddActive) return;            // ← blokkeert poly-punten tijdens POI plaatsen
  poly.push(toXY(ev));
  draw();
});

// (optioneel) dubbelklik blijft gewoon voor poly sluiten/refreshen
els.cvs.addEventListener('dblclick', draw);

// ➋ POI plaatsen: laat je bestaande click-handler staan
// (of voeg deze toe als je ‘m wil samenvoegen in één plek)
els.cvs.addEventListener('click', ev => {
  if (!poiAddActive || !W || !H) return;
  const [x, y] = toXY(ev);
  const label = (els.poiLabel.value || '').trim();
  const required = Math.max(1, parseInt(els.poiReq.value || '1', 10));
  const radius01 = Math.max(0.002, parseFloat(els.poiR01.value || '0.02'));
  if (!label) { alert('Vul expected label in'); return; }
  poiItems.push({ x01: x / W, y01: y / H, expected_label: label, required, radius01 });
  renderPOIList();
  draw();                               // ← opnieuw tekenen zodat de box meteen zichtbaar wordt
});

async function uploadPickedFileIfAny() {
  const f = els.file?.files?.[0];
  if (!f) return null;
  const fd = new FormData();
  fd.append('file', f, f.name);
  fd.append('name', f.name);
  const r = await fetch('/api/trainer/upload-image', { method: 'POST', body: fd });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || !data?.url) throw new Error(data?.error || 'Upload mislukt');
  return { image_url: data.url, image_name: f.name };
}


/* ---------- opslaan contour ---------- */
els.save.onclick = async () => {
  if (poly.length < 3) return;
  const name = els.name.value.trim();
  const type_key = els.typeKey.value.trim();
  const product_id = (els.product?.value || '') || null;
  const polygon01 = poly.map(([x, y]) => [x / W, y / H].map(v => Math.max(0, Math.min(1, v))));

  // ← probeer foto te uploaden (indien gekozen)
  let imgMeta = null;
  try { imgMeta = await uploadPickedFileIfAny(); } catch (e) { console.warn(e); /* niet fataal */ }

  const body = {
    name, type_key, width: W, height: H, polygon01, product_id,
    ...(imgMeta || {})                 // plakt image_url & image_name erbij
  };

  try {
    const r = await fetch('/api/contours', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!r.ok) throw new Error(await r.text());
    poly = [];
    draw();
    await loadList();
  } catch (e) {
    console.error(e); alert('Opslaan mislukt');
  }
};


// Undo laatste poly-punt
els.undo.onclick = () => {
  if (poiAddActive) return;            // optioneel: undo alleen voor poly, niet voor POI
  if (!poly.length) return;
  poly.pop();
  draw();
};

// Hele polygoon verwijderen
els.clear.onclick = () => {
  if (poiAddActive) return;            // optioneel
  if (!poly.length) return;
  poly = [];
  draw();
};



/* ---------- config ---------- */
els.applyCfg.onclick = async () => {
  const body = {};
  if (els.cls.value.trim()) body.contour_match_class = els.cls.value.trim();
  if (els.thr.value.trim()) body.contour_match_iop = parseFloat(els.thr.value);
  if (!Object.keys(body).length) return;
  try {
    await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    alert('Instellingen opgeslagen');
  } catch (e) {
    console.error(e);
    alert('Opslaan mislukt');
  }
};

/* ---------- init ---------- */
els.refresh.addEventListener('click', loadList);
els.search.addEventListener('input', applySearchAndRender);

(async function init() {
  try {
    const cfg = await (await fetch('/api/config')).json();
    els.cls.value = cfg?.config?.contour_match_class ?? 'hole';
    els.thr.value = cfg?.config?.contour_match_iop ?? 0.60;
  } catch (e) { console.warn('config ophalen mislukt', e); }
  await loadProducts();
  await loadList();
})();