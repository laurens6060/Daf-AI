// ------- Navbar active (zorgt dat de huidige pagina gemarkeerd wordt in de navigatie) -------
document.querySelectorAll('.nav-link').forEach(a => {
  if (a.getAttribute('href') === window.location.pathname) a.classList.add('active');
});

// ----- DOM referenties -----
// Elementen uit de HTML ophalen zodat we ze later kunnen manipuleren
const typeSel = document.getElementById('typeSel');           // dropdown voor as-type selectie
const typeCodeEl = document.getElementById('typeCode');       // toont huidig geselecteerde as-type code

const ordersTblBody = document.getElementById('ordersTbl').querySelector('tbody'); // body van orders tabel

const subsHead = document.getElementById('subsHead');         // tabelkop voor subtypes
const subsBody = document.getElementById('subsBody');         // tabelbody voor subtypes

const propEditor = document.getElementById('propEditor');     // editor voor properties (zichtbaar/invisible toggle)
const propInputs = document.getElementById('propInputs');     // container waar property-rijen komen
const addPropBtn = document.getElementById('addPropBtn');     // knop om een nieuwe property toe te voegen

const subName = document.getElementById('subName');           // input veld voor subtype naam
const subId = document.getElementById('subId');               // hidden input voor subtype ID (bij edit)
const createBtn = document.getElementById('createBtn');       // knop om een nieuw subtype aan te maken
const updateBtn = document.getElementById('updateBtn');       // knop om bestaand subtype op te slaan
const cancelEditBtn = document.getElementById('cancelEditBtn'); // knop om edit-modus te annuleren

// ----- Client state -----
// Houdt de huidig geselecteerde type en subtypes bij in JS zodat we UI makkelijk kunnen updaten
let currentType = null;      // e.g. "1344"
let currentColumns = [];     // dynamische kolommen (alle unieke property keys)
let currentSubtypes = [];    // subtypes voor huidig type

// ----- Fetch helpers -----
// Kleine wrappers rond fetch voor GET/POST/PUT/DELETE
const GET = (url) => fetch(url).then(r => r.json());
const POST = (url, body) => fetch(url, {
  method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
}).then(r => r.json());
const PUT = (url, body) => fetch(url, {
  method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
}).then(r => r.json());
const DELETE_REQ = (url) => fetch(url, {method:'DELETE'}).then(r => r.json());

// =============== Property editor helpers ===============
// Voeg een nieuwe lege property-rij toe aan de editor
function addEmptyPropRow(key = "", val = "") {
  const row = document.createElement('div');
  row.className = 'd-flex align-items-center gap-2';

  // Input voor property-naam
  const nameInp = document.createElement('input');
  nameInp.type = 'text';
  nameInp.placeholder = 'property';
  nameInp.className = 'form-control form-control-sm';
  nameInp.style.maxWidth = '180px';
  nameInp.value = key;

  // Input voor property-waarde (nummer)
  const valueInp = document.createElement('input');
  valueInp.type = 'number';
  valueInp.min = '0';
  valueInp.step = '1';
  valueInp.placeholder = '0';
  valueInp.className = 'form-control form-control-sm';
  valueInp.style.maxWidth = '120px';
  valueInp.value = val === "" ? "" : String(val);

  // Verwijder-knop voor deze rij
  const rmBtn = document.createElement('button');
  rmBtn.type = 'button';
  rmBtn.className = 'btn btn-outline-danger btn-sm';
  rmBtn.textContent = 'Remove';
  rmBtn.onclick = () => row.remove();

  // Alles toevoegen aan de rij en dan aan de editor
  row.appendChild(nameInp);
  row.appendChild(valueInp);
  row.appendChild(rmBtn);
  propInputs.appendChild(row);
}

// Haal alle property key/value pairs uit de editor op
function gatherValuesFromForm() {
  const values = {};
  propInputs.querySelectorAll('.d-flex.align-items-center').forEach(row => {
    const [nameInp, valueInp] = row.querySelectorAll('input');
    const key = (nameInp.value || '').trim();
    if (!key) return;
    const raw = valueInp.value;
    const num = raw === '' ? 0 : parseInt(raw, 10);
    values[key] = Number.isNaN(num) ? 0 : num;
  });
  return values;
}

// ------- UI mode toggles -------
// Wissel naar "create" modus (verberg property editor)
function showCreate() {
  createBtn.classList.remove('d-none');
  updateBtn.classList.add('d-none');
  cancelEditBtn.classList.add('d-none');
  propEditor.classList.add('d-none');
}

// Wissel naar "edit" modus (toon property editor en juiste knoppen)
function showEdit() {
  createBtn.classList.add('d-none');
  updateBtn.classList.remove('d-none');
  cancelEditBtn.classList.remove('d-none');
  propEditor.classList.remove('d-none');
}

// Formulier leegmaken en terug naar create-modus
function clearForm() {
  subId.value = "";
  subName.value = "";
  propInputs.innerHTML = "";
  showCreate();
}

// Formulier vullen met een bestaand subtype en naar edit-modus gaan
function fillFormFromSubtype(st) {
  subId.value = st.id;
  subName.value = st.name;
  propInputs.innerHTML = "";
  const entries = Object.entries(st.values || {});
  if (!entries.length) addEmptyPropRow();  // voeg lege rij toe indien geen properties
  for (const [k, v] of entries) addEmptyPropRow(k, v);
  showEdit();
}

// =============== Table rendering ===============
// Tabel header opnieuw tekenen (kolommen dynamisch op basis van properties)
function renderSubsHead() {
  subsHead.innerHTML = "";
  const tr = document.createElement('tr');
  const thName = document.createElement('th'); thName.textContent = 'Subtype';
  tr.appendChild(thName);
  for (const c of currentColumns) {
    const th = document.createElement('th'); th.textContent = c;
    tr.appendChild(th);
  }
  const thAct = document.createElement('th'); thAct.textContent = 'Actions';
  tr.appendChild(thAct);
  subsHead.appendChild(tr);
}

// Body van subtypes-tabel renderen
function renderSubsBody() {
  subsBody.innerHTML = "";
  if (!currentSubtypes.length) {
    // Geen subtypes: toon melding
    const tr = document.createElement('tr');
    const td = document.createElement('td'); td.colSpan = currentColumns.length + 2;
    td.className = 'text-muted'; td.textContent = 'No subtypes';
    tr.appendChild(td);
    subsBody.appendChild(tr);
    return;
  }

  // Voor elke subtype een rij tekenen
  for (const st of currentSubtypes) {
    const tr = document.createElement('tr');

    const tdName = document.createElement('td');
    tdName.textContent = st.name;
    tr.appendChild(tdName);

    // Kolommen voor alle properties
    for (const c of currentColumns) {
      const td = document.createElement('td');
      td.textContent = st.values?.[c] ?? "";
      tr.appendChild(td);
    }

    // Actieknoppen (edit / delete)
    const tdAct = document.createElement('td');
    const editBtn = document.createElement('button');
    editBtn.className = 'btn btn-success btn-sm me-2';
    editBtn.textContent = 'Edit';
    editBtn.onclick = () => fillFormFromSubtype(st);

    const delBtn = document.createElement('button');
    delBtn.className = 'btn btn-danger btn-sm';
    delBtn.textContent = 'Delete';
    delBtn.onclick = async () => {
      if (!confirm(`Delete subtype "${st.name}"?`)) return;
      await DELETE_REQ(`/mes/axle-types/${encodeURIComponent(currentType)}/subtypes/${encodeURIComponent(st.id)}`);
      await loadType(currentType); // herlaad tabel na verwijderen
      clearForm();
    };

    tdAct.appendChild(editBtn);
    tdAct.appendChild(delBtn);
    tr.appendChild(tdAct);

    subsBody.appendChild(tr);
  }
}

// Orders tabel links vullen
async function loadOrders() {
  const rows = await GET('/mes/orders');
  ordersTblBody.innerHTML = "";
  for (const o of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${o.order_id}</td><td>${o.axle_type}</td><td>${o.axle_subtype}</td><td>${o.qty}</td>`;
    ordersTblBody.appendChild(tr);
  }
}

// Een enkel as-type ophalen en tonen
async function loadType(code) {
  const data = await GET(`/mes/axle-types/${encodeURIComponent(code)}`);
  currentType = code;
  currentSubtypes = data.subtypes || [];

  // Gebruik kolommen van server (indien aanwezig) anders zelf unie maken
  if (Array.isArray(data.columns)) {
    currentColumns = data.columns;
  } else {
    const colset = new Set();
    for (const st of currentSubtypes) {
      Object.keys(st.values || {}).forEach(k => colset.add(k));
    }
    currentColumns = Array.from(colset).sort();
  }

  typeCodeEl.textContent = currentType;
  renderSubsHead();
  renderSubsBody();

  // Reset formulier naar "idle" toestand
  clearForm();
}

// Dropdown vullen met beschikbare axle types
async function loadTypes() {
  const types = await GET('/mes/axle-types');
  typeSel.innerHTML = "";
  for (const t of types) {
    const opt = document.createElement('option');
    opt.value = t.code; opt.textContent = t.code;
    typeSel.appendChild(opt);
  }
  if (types.length) {
    typeSel.value = types[0].code;
    await loadType(types[0].code);
  }
}

// ----- Event listeners -----
// Type switch in dropdown
typeSel.addEventListener('change', async () => {
  await loadType(typeSel.value);
});

// Knop om nieuwe property toe te voegen
addPropBtn.addEventListener('click', () => addEmptyPropRow());

// Create-knop: nieuw subtype aanmaken
createBtn.addEventListener('click', async () => {
  if (!currentType) return;
  const name = subName.value.trim();
  if (!name) { alert('Please enter a subtype name'); return; }
  const values = {}; // nieuwe subtype start leeg
  const res = await POST(`/mes/axle-types/${encodeURIComponent(currentType)}/subtypes`, { name, values });
  if (res.error) { alert(res.error); return; }
  await loadType(currentType);
  clearForm();
});

// Update-knop: bestaand subtype opslaan
updateBtn.addEventListener('click', async () => {
  if (!currentType) return;
  const id = subId.value;
  if (!id) return;
  const name = subName.value.trim();
  const values = gatherValuesFromForm();  // properties ophalen uit formulier
  const res = await PUT(`/mes/axle-types/${encodeURIComponent(currentType)}/subtypes/${encodeURIComponent(id)}`, { name, values });
  if (res.error) { alert(res.error); return; }
  await loadType(currentType);
  clearForm();
});

// Annuleer edit-modus
cancelEditBtn.addEventListener('click', clearForm);

// ----- Initieel laden van data -----
(async () => {
  await loadTypes();
  await loadOrders();
})();
