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

const ctx = () => els.cvs.getContext('2d');

let _products = [];
let _prodById = new Map();

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
          <button class="btn btn-sm btn-outline-danger del">Verwijderen</button>
        </div>
      </div>
    `;

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
function draw() {
  const c = ctx();
  els.cvs.width = W; els.cvs.height = H;
  c.clearRect(0, 0, W, H);
  if (poly.length) {
    c.beginPath();
    c.moveTo(poly[0][0], poly[0][1]);
    for (let i = 1; i < poly.length; i++) c.lineTo(poly[i][0], poly[i][1]);
    c.strokeStyle = '#ffcc38'; c.lineWidth = Math.max(2, Math.round(W / 400)); c.setLineDash([10, 6]); c.stroke();
    c.setLineDash([]);
    c.fillStyle = '#ffcc38';
    for (const p of poly) { c.beginPath(); c.arc(p[0], p[1], Math.max(3, W / 300), 0, Math.PI * 2); c.fill(); }
  }
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
els.cvs.addEventListener('mousedown', ev => {
  if (!W || !H) return;
  poly.push(toXY(ev));
  draw();
});
els.cvs.addEventListener('dblclick', draw);
els.undo.onclick = () => { poly.pop(); draw(); };
els.clear.onclick = () => { poly = []; draw(); };

/* ---------- opslaan contour ---------- */
els.save.onclick = async () => {
  if (poly.length < 3) return;
  const name = els.name.value.trim();
  const type_key = els.typeKey.value.trim();
  const product_id = (els.product?.value || '') || null; // optioneel
  const polygon01 = poly.map(([x, y]) => [x / W, y / H].map(v => Math.max(0, Math.min(1, v))));

  const body = { name, type_key, width: W, height: H, polygon01, product_id };

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
    console.error(e);
    alert('Opslaan mislukt');
  }
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