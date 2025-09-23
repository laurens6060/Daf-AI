// ========================================================================
// Products/Collections UI
// - Beheert links de lijst van collecties (filter) + CRUD (create/update/delete)
// - Toont rechts een grid met producten en dynamische kolommen o.b.v. properties
// - Eenvoudige product-editor (naam, collectie, vrije properties als key/value)
// - Communiceert met backend routes:
//     * GET/POST /collections, PUT/DELETE /collections/{cid}
//     * GET/POST /products,   PUT/DELETE /products/{pid}
// ========================================================================

// ------- DOM referenties -------
// Alle relevante elementen 1x opzoeken, later enkel hergebruiken.
const els = {
    // Sidebar (collecties + "nieuwe collectie")
    collections: document.getElementById('collections'),
    newCollection: document.getElementById('newCollection'),

    // Header (boven het grid; toont huidige filter)
    currentCollectionName: document.getElementById('currentCollectionName'),
    collectionHint: document.getElementById('collectionHint'),

    // Grid + editor (producttabel + form rechts)
    gridHead: document.querySelector('#grid thead'),
    gridBody: document.querySelector('#grid tbody'),
    pName: document.getElementById('pName'),                 // input: productnaam
    pCollection: document.getElementById('pCollection'),     // select: collectie van product
    propList: document.getElementById('propList'),           // container: lijst van property-rijen
    addProp: document.getElementById('addProp'),             // knop: property-rij toevoegen
    saveBtn: document.getElementById('saveBtn'),             // knop: product opslaan (create/update)
    resetBtn: document.getElementById('resetBtn'),           // knop: editor leegmaken
    newProduct: document.getElementById('newProduct'),       // knop: nieuw product aanmaken (editor reset)
    reloadBtn: document.getElementById('reloadBtn'),         // knop: refresh productlijst

    // Logging (diagnostiek)
    log: document.getElementById('log'),
    clearLog: document.getElementById('clearLog'),
};

// ------- In-memory state -------
// Centrale status voor deze pagina; we vermijden globale losse variabelen.
let state = {
    collections: [],        // lijst van {id, name, ...}
    items: [],              // lijst van producten (server payload)
    editId: null,           // null = nieuw product; anders: product-id in bewerking
    columns: [],            // kolomnamen voor het grid (name + dynamische properties)
    selectedCollectionId: null, // huidige filter (null = alle producten)
};

// ------- Logging helpers -------
function log(msg) {
    // Voeg een regel toe aan het logvlak en scroll naar de onderkant
    els.log.textContent += `\n${msg}`;
    els.log.scrollTop = els.log.scrollHeight;
}
els.clearLog.onclick = () => els.log.textContent = '';

// ------- Type parser voor property-waarden -------
// Zet "true"/"false" om naar booleans, numerieke strings naar getallen,
// lege string blijft lege string. Alles anders blijft string.
function parseVal(text) {
    const t = (text ?? '').trim();
    if (t === '') return '';
    if (t === 'true') return true;
    if (t === 'false') return false;
    const n = Number(t);
    if (!isNaN(n) && t.match(/^[+-]?\d+(\.\d+)?$/)) return n;
    return t;
}

// ------- UI: property-rij (key/value) maken -------
// Een compacte rij met input voor sleutel en waarde + delete-knop.
// Wordt gebruikt in de product-editor (vrije properties).
function propRow(key = '', val = '') {
    const wrap = document.createElement('div');
    wrap.className = 'd-flex prop-row align-items-center gap-2 my-1';
    wrap.innerHTML = `
        <input class="form-control form-control-sm key" placeholder="eigenschap (bv. diameter)" value="${key}">
        <input class="form-control form-control-sm val" placeholder="waarde (bv. 320 of staal)" value="${val}">
        <button class="btn btn-sm btn-outline-danger del" title="Verwijder">&times;</button>
    `;
    // Verwijder alleen deze rij, laat de rest ongemoeid
    wrap.querySelector('.del').onclick = () => wrap.remove();
    return wrap;
}

// ========================================================================
// Collections UI (links): renderen, selecteren, CRUD
// ========================================================================
function renderCollections() {
    // Sidebar links leegmaken en volledig opnieuw opbouwen
    els.collections.innerHTML = '';

    // "Alle producten"-knop (geen filter)
    const all = document.createElement('button');
    all.className = `list-group-item list-group-item-action collection-item text-dark ${state.selectedCollectionId ? '' : 'active'}`;
    all.textContent = 'Alle producten';
    all.onclick = () => selectCollection(null);
    els.collections.appendChild(all);

    // Voor elke collectie een item met: kleurbol, (optioneel) icoon + naam, en bewerk/verwijder knoppen
    state.collections.forEach(c => {
        const btn = document.createElement('div');
        btn.className = `list-group-item d-flex justify-content-between align-items-center collection-item text-dark ${state.selectedCollectionId === c.id ? 'active' : ''}`;
        btn.innerHTML = `
            <div class="d-flex align-items-center gap-2">
                <span style="width:10px;height:10px;border-radius:50%;background:${c.color || ''};display:inline-block;"></span>
                <span>${c.icon ? c.icon + ' ' : ''}<strong>${c.name}</strong></span>
            </div>
            <div>
                <button class="btn btn-sm btn-outline-primary me-1 edit">Bewerk</button>
                <button class="btn btn-sm btn-outline-danger del">X</button>
            </div>
        `;

        // BEWERK: enkel naam aanpassen (server: PUT /collections/{cid} {name})
        btn.querySelector('.edit').onclick = async () => {
            const newName = prompt('Nieuwe naam:', c.name);
            if (newName == null) return; // gebruiker cancelde
            try {
                const r = await fetch(`/collections/${c.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: newName })
                });
                const data = await r.json(); if (!r.ok) throw new Error(data?.error || r.status);
                await loadCollections();
                renderCollections();
                loadProducts(); // lijst rechts herbouwen i.v.m. naamwijziging
            } catch (e) { log(`Bewerken collectie fout: ${e.message}`); }
        };

        // VERWIJDER: collectie verwijderen; server koppelt producten los (collection_id=null)
        btn.querySelector('.del').onclick = async () => {
            if (!confirm(`Verwijder collectie "${c.name}"? Producten worden losgekoppeld.`)) return;
            try {
                const r = await fetch(`/collections/${c.id}`, { method: 'DELETE' });
                const data = await r.json(); if (!r.ok) throw new Error(data?.error || r.status);
                if (state.selectedCollectionId === c.id) state.selectedCollectionId = null;
                await loadCollections();
                renderCollections();
                await loadProducts(); // refresh grid
            } catch (e) { log(`Verwijderen collectie fout: ${e.message}`); }
        };

        // Klik op het item zelf = selecteer als filter (behalve op knoppen)
        btn.onclick = (ev) => {
            if (ev.target.closest('button')) return; // klik was op edit/del
            selectCollection(c.id);
        };

        els.collections.appendChild(btn);
    });

    // Editor-select (product->collectie) vullen met huidige collecties
    els.pCollection.innerHTML = `<option value="">(geen)</option>` +
        state.collections.map(c => `<option value="${c.id}">${c.name}</option>`).join('');
}

// Nieuwe collectie aanmaken (prompt -> POST /collections)
async function newCollection() {
    const name = prompt('Naam van collectie (bv. Flens):');
    if (!name) return;
    try {
        const r = await fetch('/collections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await r.json(); if (!r.ok) throw new Error(data?.error || r.status);
        await loadCollections();
        renderCollections();
    } catch (e) { log(`Aanmaken collectie fout: ${e.message}`); }
}
els.newCollection.onclick = newCollection;

// Huidige filter aanpassen en UI bijwerken
function selectCollection(cid) {
    state.selectedCollectionId = cid;
    const c = state.collections.find(x => x.id === cid);

    // Header boven het grid aanpassen
    els.currentCollectionName.textContent = c ? c.name : 'Alle producten';
    els.collectionHint.textContent = c
        ? `Toont alleen producten in "${c.name}".`
        : 'Selecteer links een collectie om te filteren.';

    // Sidebar (active class) en grid hertekenen
    renderCollections();
    loadProducts();
}

// ========================================================================
// Product editor (rechts): form-gedrag en data verzamelen
// ========================================================================

// Property toevoegen/leegmaken/nieuw product beginnen/refresh
els.addProp.onclick   = () => els.propList.appendChild(propRow());
els.resetBtn.onclick  = () => setEditor();
els.newProduct.onclick= () => setEditor();
els.reloadBtn.onclick = () => loadProducts();

/**
 * Zet de editor in "nieuw" (item=null) of "bewerken" (item gevuld).
 * - Bij nieuw: editId=null, collectie default = huidige filter
 * - Bij bewerken: vul alle velden + properties in
 */
function setEditor(item = null) {
    if (!item) {
        state.editId = null;
        els.pName.value = '';
        els.pCollection.value = state.selectedCollectionId || '';
        els.propList.innerHTML = '';
        return;
    }
    state.editId = item.id;
    els.pName.value = item.name || '';
    els.pCollection.value = item.collection_id || '';
    els.propList.innerHTML = '';
    Object.entries(item.properties || {}).forEach(([k, v]) => {
        els.propList.appendChild(propRow(k, String(v)));
    });
}

/**
 * Leest de editor-waarden uit de DOM en normaliseert naar payload.
 * - properties: key/value met parseVal (booleans/nummers/strings)
 */
function gatherEditor() {
    const name = (els.pName.value || '').trim();
    const collection_id = els.pCollection.value || null;
    const rows = [...els.propList.querySelectorAll('.prop-row')];
    const properties = {};
    rows.forEach(r => {
        const k = r.querySelector('.key').value.trim();
        const v = parseVal(r.querySelector('.val').value);
        if (k) properties[k] = v;
    });
    return { name, collection_id, properties };
}

// Opslaan (create of update) op basis van state.editId
els.saveBtn.onclick = async () => {
    const { name, collection_id, properties } = gatherEditor();
    if (!name) { alert('Naam is verplicht.'); return; }

    try {
        if (state.editId) {
            // Update bestaand product
            const r = await fetch(`/products/${state.editId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, collection_id, properties })
            });
            const data = await r.json(); if (!r.ok) throw new Error(data?.error || r.status);
            log(`Aangepast: ${data.name}`);
        } else {
            // Maak nieuw product
            const r = await fetch('/products', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, collection_id, properties })
            });
            const data = await r.json(); if (!r.ok) throw new Error(data?.error || r.status);
            log(`Aangemaakt: ${data.name}`);
        }

        // Editor resetten en grid verversen
        setEditor(null);
        await loadProducts();
    } catch (e) {
        log(`Opslaan mislukt: ${e.message}`);
    }
};

// ========================================================================
// Grid (productentabel): kolommen bepalen en renderen
// ========================================================================

/**
 * Bepaalt dynamische kolomnamen:
 * - Altijd eerste kolom 'name'
 * - Daarna alfabetische sortering van alle unieke property-sleutels
 */
function computeColumns(items) {
    const keys = new Set();
    items.forEach(it => Object.keys(it.properties || {}).forEach(k => keys.add(k)));
    return ['name', ...Array.from(keys).sort()];
}

/**
 * Bouwt de tabel op:
 * - Kolomkoppen o.b.v. state.columns
 * - Rij per item, properties worden in de juiste kolom getoond indien aanwezig
 * - Acties: Bewerk/Verwijder
 */
function renderTable() {
    const cols = state.columns;

    // Kopregel
    els.gridHead.innerHTML = `
        <tr>
            ${cols.map(c => `<th>${c === 'name' ? 'Product' : c}</th>`).join('')}
            <th>Acties</th>
        </tr>
    `;

    // Body
    els.gridBody.innerHTML = '';
    state.items.forEach(it => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            ${cols.map(c => {
                if (c === 'name') return `<td><strong>${it.name || ''}</strong></td>`;
                const v = it.properties?.[c];
                return `<td>${v === undefined ? '' : String(v)}</td>`;
            }).join('')}
            <td class="text-nowrap">
                <button class="btn btn-sm btn-outline-primary me-1 edit">Bewerk</button>
                <button class="btn btn-sm btn-outline-danger del">Verwijder</button>
            </td>
        `;

        // Bewerk: laad item in de editor
        tr.querySelector('.edit').onclick = () => setEditor(it);

        // Verwijder: DELETE /products/{pid} en daarna refresh
        tr.querySelector('.del').onclick = async () => {
            if (!confirm(`Verwijder "${it.name}"?`)) return;
            try {
                const r = await fetch(`/products/${it.id}`, { method: 'DELETE' });
                const data = await r.json(); if (!r.ok) throw new Error(data?.error || r.status);
                log(`Verwijderd: ${it.name}`);
                await loadProducts();
            } catch (e) { log(`Verwijderen mislukt: ${e.message}`); }
        };

        els.gridBody.appendChild(tr);
    });
}

// ========================================================================
// Data laden (fetch) en state bijwerken
// ========================================================================

/**
 * Haal collecties op en schrijf naar state.collections
 */
async function loadCollections() {
    try {
        const r = await fetch('/collections');
        const data = await r.json();
        if (!Array.isArray(data)) throw new Error('Collections payload ongeldig');
        state.collections = data;
    } catch (e) {
        log(`Laden collecties mislukt: ${e.message}`);
    }
}

/**
 * Haal producten (optioneel gefilterd op selectedCollectionId) op en render tabel.
 * - Bepaalt ook de dynamische kolommen o.b.v. properties.
 */
async function loadProducts() {
    try {
        const q = state.selectedCollectionId ? `?collection_id=${encodeURIComponent(state.selectedCollectionId)}` : '';
        const r = await fetch(`/products${q}`);
        const data = await r.json();
        if (!Array.isArray(data)) throw new Error('Products payload ongeldig');

        state.items = data;
        state.columns = computeColumns(state.items);
        renderTable();
    } catch (e) {
        log(`Laden producten mislukt: ${e.message}`);
    }
}

// ========================================================================
// Init: startsequentie bij pagina-load
// ========================================================================
(async function init() {
    await loadCollections();  // eerst collecties (ivm pCollection dropdown)
    renderCollections();      // sidebar & editor dropdown opbouwen
    await loadProducts();     // products grid opbouwen
})();
