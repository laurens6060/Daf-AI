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

const carouselEls = {
  pick: document.getElementById('carPick'),
  file: document.getElementById('carFile'),
  inner: document.getElementById('carInner'),
  indicators: document.getElementById('carIndicators'),
  carousel: document.getElementById('contourCarousel'),
  status: document.getElementById('carStatus'),
  log: document.getElementById('carLog'),
  btnCheck: document.getElementById('contourCheck'),
  btnCheckAll: document.getElementById('contourCheckAll'),
  ref: document.getElementById('contourRef'),
  thrScore: document.getElementById('contourScore'),
  thrIoU: document.getElementById('contourIoU'),
  progWrap: document.getElementById('carProgress'),
  progBar: document.querySelector('#carProgress .progress-bar'),
  count: document.getElementById('carCount'),
};

const carouselState = {
  items: [] // [{name, url, imgEl, badgeEl, width, height}]
};

function carLog(msg) {
  if (!carouselEls.log) return;
  carouselEls.log.textContent += (carouselEls.log.textContent ? '\n' : '') + msg;
  carouselEls.log.scrollTop = carouselEls.log.scrollHeight;
}
function carStatus(text, cls='text-bg-secondary') {
  if (!carouselEls.status) return;
  carouselEls.status.className = 'badge ' + cls;
  carouselEls.status.textContent = text;
}
function setProgress(ratio, note='') {
  if (!carouselEls.progWrap || !carouselEls.progBar) return;
  carouselEls.projawrap?.classList?.remove('d-none'); // safeguard
}
function showProgress(ratio, note) {
  if (!carouselEls.progWrap || !carouselEls.progBar) return;
  carouselEls.progWrap.classList.remove('d-none');
  const pct = Math.max(0, Math.min(100, Math.round(ratio * 100)));
  carouselEls.progBar.style.width = pct + '%';
  carouselEls.count.textContent = note || '';
}
function hideProgress() {
  if (!carouselEls.progWrap || !carouselEls.progBar) return;
  carouselEls.progBar.style.width = '0%';
  carouselEls.progWrap.classList.add('d-none');
  carouselEls.count.textContent = '';
}

function addCarouselImage(name, url) {
  const idx = carouselState.items.length;

  // indicator
  const dot = document.createElement('button');
  dot.type = 'button';
  dot.setAttribute('data-bs-target', '#contourCarousel');
  dot.setAttribute('data-bs-slide-to', String(idx));
  dot.setAttribute('aria-label', `Slide ${idx+1}`);
  if (idx === 0) { dot.className = 'active'; dot.setAttribute('aria-current','true'); }
  carouselEls.indicators.appendChild(dot);

  // slide
  const slide = document.createElement('div');
  slide.className = 'carousel-item' + (idx===0 ? ' active':'');
  const wrap = document.createElement('div');
  wrap.className = 'position-relative';
  const img = new Image();
  img.className = 'd-block w-100';
  img.src = url;
  img.alt = name;

  const badge = document.createElement('span');
  badge.className = 'badge position-absolute top-0 start-0 m-2 text-bg-secondary';
  badge.textContent = '…';

  wrap.appendChild(img);
  wrap.appendChild(badge);
  slide.appendChild(wrap);
  carouselEls.inner.appendChild(slide);

  const item = { name, url, imgEl: img, badgeEl: badge, width: 0, height: 0 };
  carouselState[Symbol.for('lastAdded')] = item;
  carouselState.items.push(item);

  img.onload = () => {
    item.width = img.naturalWidth || img.width;
    item.height = img.naturalHeight || img.height;
  };
}

function getActiveIndex() {
  const slides = Array.from(carouselEls.inner.querySelectorAll('.carousel-item'));
  return Math.max(0, slides.findIndex(s => s.classList.contains('active')));
}
function getActiveItem() {
  const i = getActiveIndex();
  return carouselState.items[i];
}

async function contourMatchItem(item) {
  if (!item || !item.imgEl || !item.width) {
    carLog('⚠️ Geen geldige afbeelding om te matchen.');
    return { ok:false };
  }
  const ref = (carouselEls.ref?.value || '').trim();
  const scoreMax = parseFloat((carouselEls.thrScore?.value || '0.04').replace(',','.')) || 0.04;
  const iouMin   = parseFloat((carouselEls.thrIoU?.value   || '0.60').replace(',','.')) || 0.60;

  // render naar offscreen canvas om JPEG te maken
  const off = document.createElement('canvas');
  off.width = item.width; off.height = item.height;
  const ctx = off.getContext('2d');
  ctx.drawImage(item.imgEl, 0, 0, off.width, off.height);
  const blob = await new Promise(res => off.toBlob(res, 'image/jpeg', 0.95));

  const fd = new FormData();
  fd.append('file', new File([blob], item.name || 'image.jpg'));
  if (ref) fd.append('ref_name', ref);
  fd.append('score_thresh', String(scoreMax));
  fd.append('iou_thresh', String(iouMin));

  let resp, raw, data;
  try {
    resp = await fetch('/api/contour-match', { method: 'POST', body: fd });
    raw = await resp.text();
    try { data = raw ? JSON.parse(raw) : {}; } catch { data = null; }
  } catch (e) {
    carStatus('Netwerkfout', 'text-bg-danger');
    carLog(`❌ Netwerkfout: ${e}`);
    item.badgeEl.className = 'badge position-absolute top-0 start-0 m-2 text-bg-danger';
    item.badgeEl.textContent = 'ERR';
    return { ok:false };
  }

  if (!resp.ok || !data || !data.best) {
    carStatus('Fout', 'text-bg-danger');
    carLog(`❌ Serverfout: ${resp?.status} ${raw?.slice(0,200) || ''}`);
    item.badgeEl.className = 'badge position-absolute top-0 start-0 m-2 text-bg-danger';
    item.badgeEl.textContent = 'ERR';
    return { ok:false };
  }

  const b = data.best;
  const ok = !!b.match;
  const label = ok ? (b.type_key ? `${b.type_key}` : 'OK') : 'geen match';
  item.badgeEl.className = 'badge position-absolute top-0 start-0 m-2 ' + (ok ? 'text-bg-success' : 'text-bg-warning');
  item.badgeEl.textContent = label;

  carLog(`[Contour] ${ok ? 'MATCH' : 'GEEN match'} • type=${b.type_key || '?'} • ref=${b.ref} • score=${(b.score??0).toFixed(3)} • IoU=${(b.iou??0).toFixed(2)}`);
  return { ok, best: b, all: data.all || [] };
}

async function contourMatchActive() {
  const item = getChildrenSafe(getActiveItem);
  const cur = getActiveItem();
  if (!cur) { carLog('⚠️ Geen actieve slide.'); return; }
  carStatus('Bezig…', 'text-bg-info');
  await contourMatchItem(cur);
  carStatus('Gereed', 'text-bg-secondary');
}

async function contourMatchAllSlides() {
  if (!carouselState.items.length) { carLog('⚠️ Voeg eerst afbeeldingen toe.'); return; }
  carStatus('Batch…', 'text-bg-info');
  carouselEls.btnCheckAll?.setAttribute('disabled', 'true');

  let okCount = 0;
  for (let i=0; i<carouselState.items.length; i++) {
    const it = carouselState.items[i];
    const res = await contourMatchItem(it);
    if (res.ok) okCount++;
    showProgress((i+1)/carouselState.items.length, `(${i+1}/${carouselState.items.length}) matches: ${okCount}`);
    await new Promise(r => setTimeout(r, 30)); // UI ademruimte
  }
  hideProgress();
  carStatus(`Klaar • ${okCount}/${carouselState.items.length} match(es)`, okCount ? 'text-bg-success' : 'text-bg-warning');
  carouselEls.btnCheckAll?.removeAttribute('disabled');
}

carouselEls.pick?.addEventListener('click', () => carouselEls.file?.click());
carouselEls.file?.addEventListener('change', (e) => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;
  files.forEach(f => {
    const url = URL.createObjectURL(f);
    addCarouselImage(f.name, url);
  });
  carouselEls.count.textContent = `${carouselState.items.length} afbeelding(en) geladen`;
  carLog(`+ ${files.length} afbeelding(en) toegevoegd.`);
});

carouselEls.btnCheck?.addEventListener('click', contourMatchActive);
carouselEls.btnCheckAll?.addEventListener('click', contourMatchAllSlides);

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

// === Ref-contours upload ===
const refEls = {
  pick: document.getElementById('refPick'),
  file: document.getElementById('refFile'),
  name: document.getElementById('refName'),
  msg:  document.getElementById('refUploadMsg'),
};

function _flashRefMsg(text, ok=true) {
  if (!refEls.msg) return;
  refEls.msg.textContent = text;
  refEls.msg.className = 'form-text ' + (ok ? 'text-success' : 'text-danger');
  setTimeout(() => { refEls.msg.textContent = ''; refEls.msg.className = 'form-text'; }, 3000);
}

refEls.pick?.addEventListener('click', () => refEls.file?.click());

refEls.file?.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;

  try {
    for (const f of files) {
      const fd = new FormData();
      // Naam: veldje overschrijft bestandsnaam (zonder extensie). Handig voor T9/T13.
      const nm = (refEls.name?.value || '').trim();
      if (nm) fd.append('name', nm);

      fd.append('file', f, f.name);

      const r = await fetch('/api/contours/ref/upload', { method: 'POST', body: fd });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data?.error || `${r.status}`);

      // hint: wis het naamveld na één upload, zodat je per file makkelijk andere namen kan geven
      refEls.name.value = '';

      _flashRefMsg(`Geüpload: ${data.filename}`, true);
    }
  } catch (err) {
    console.error(err);
    _flashRefMsg('Upload mislukt', false);
  } finally {
    // reset input zodat dezelfde file opnieuw kan gekozen worden
    refEls.file.value = '';
  }
});




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

/* ========= Contour-match (Trainer-style) ========= */

// hulpfunctie: bbox van huidige poly (voor ROI)
function _polyBBoxPx(points) {
  if (!points?.length) return null;
  let xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const x = Math.max(0, Math.floor(Math.min(...xs)));
  const y = Math.max(0, Math.floor(Math.min(...ys)));
  const x2 = Math.min(W, Math.ceil(Math.max(...xs)));
  const y2 = Math.min(H, Math.ceil(Math.max(...ys)));
  return { x, y, w: Math.max(0, x2 - x), h: Math.max(0, y2 - y) };
}

// hulpfunctie: toon badge linksboven de canvas
function _showMatchBadge(ok, text) {
  const host = document.getElementById('matchBadge');
  if (!host) return;
  host.innerHTML = '';
  const span = document.createElement('span');
  span.className = 'badge ' + (ok ? 'text-bg-success' : 'text-bg-danger');
  span.textContent = text;
  host.appendChild(span);
}

// hulpfunctie: exporteer bronafbeelding als JPEG-blob (zonder overlays)
async function _currentImageBlob() {
  // Als user net een file gekozen heeft, gebruik die direct.
  const f = els.file?.files?.[0];
  if (f) return new Blob([await f.arrayBuffer()], { type: f.type || 'image/jpeg' });

  // Anders render wat er in <img> staat
  if (!W || !H || !els.img?.src) return null;
  const off = document.createElement('canvas');
  off.width = W; off.height = H;
  const octx = off.getContext('2d');
  octx.drawImage(els.img, 0, 0, W, H);
  return await new Promise(res => off.toBlob(res, 'image/jpeg', 0.95));
}

// ROI visueel tekenen (gestippelde rand)
function _drawRoiOverlay(bbox, ok) {
  draw(); // achtergrond opnieuw
  const c = ctx();
  if (!bbox) return;
  c.save();
  c.setLineDash([10, 6]);
  c.lineWidth = Math.max(3, Math.round(W / 400));
  c.strokeStyle = ok ? '#22e38f' : '#ef4444';
  c.strokeRect(bbox.x, bbox.y, bbox.w, bbox.h);
  c.restore();
}

// hoofdactie: stuur naar /api/contour-match
async function runContourMatch() {
  try {
    const refName = (document.getElementById('contourRef')?.value || '').trim();
    const scoreMax = parseFloat(document.getElementById('contourScore')?.value || '0.04') || 0.04;
    const iouMin = parseFloat(document.getElementById('contourIoU')?.value || '0.60') || 0.60;
    const useROI = !!document.getElementById('contourUseROI')?.checked;

    const blob = await _currentImageBlob();
    if (!blob) { alert('Geen afbeelding geladen.'); return; }

    // ROI (px) -> "x,y,w,h"
    let roiStr = '';
    let roiBox = null;
    if (useROI && poly.length >= 3) {
      roiBox = _polyBBoxPx(poly);
      if (roiBox && roiBox.w > 10 && roiBox.h > 10) {
        roiStr = `${roiBox.x},${roiBox.y},${roiBox.w},${roiBox.h}`;
      }
    }

    const fd = new FormData();
    fd.append('file', new File([blob], (els.name?.value || 'image') + '.jpg'));
    if (refName) fd.append('ref_name', refName);
    fd.append('score_thresh', String(scoreMax));
    fd.append('iou_thresh', String(iouMin));
    if (roiStr) fd.append('roi', roiStr);

    const resp = await fetch('/api/contour-match', { method: 'POST', body: fd });
    const raw = await resp.text();
    let data = {};
    try { data = raw ? JSON.parse(raw) : {}; } catch { /* laat hieronder afvangen */ }

    if (!resp.ok || !data?.best) {
      _showMatchBadge(false, 'Fout');
      console.warn('Contour-match response:', resp.status, raw);
      alert('Contour-match: serverfout of ongeldige response');
      if (roiBox) _drawRoiOverlay(roiBox, false);
      return;
    }

    const best = data.best;
    const ok = !!best.match;
    const txt = ok
      ? `Match: ${best.ref || ''} (score ${best.score?.toFixed?.(3)}, IoU ${best.iou?.toFixed?.(2)})`
      : `Geen match (beste=${best.ref || '?'}, score ${best.score?.toFixed?.(3)}, IoU ${best.iou?.toFixed?.(2)})`;

    _showMatchBadge(ok, ok ? `Match ${best.ref || ''}` : 'Geen match');
    if (roiBox) _drawRoiOverlay(roiBox, ok);
    else draw(); // refresh zodat badge zichtbaar is

    console.log('[Contour]', txt);
  } catch (e) {
    console.error(e);
    _showMatchBadge(false, 'Fout');
    alert('Contour-match mislukt.');
  }
}

// knop
document.getElementById('contourCheck')?.addEventListener('click', runContourMatch);



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