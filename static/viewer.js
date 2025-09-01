(function () {
  // Zoek de tabel-body waar de detectie-resultaten in komen
  const tbody = document.getElementById("detBody");

  // Kies het juiste protocol voor WebSocket:
  // - als de pagina via HTTPS draait, gebruik 'wss://' (secure WebSocket)
  // - anders gebruik 'ws://'
  const proto = location.protocol === "https:" ? "wss" : "ws";

  // Maak de WebSocket-verbinding naar onze backend (/ws endpoint)
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  // Zodra de verbinding open is: stuur een korte "hi" (optionele handshake/keepalive)
  ws.onopen = () => ws.send("hi");

  // Wanneer er een bericht van de server binnenkomt
  ws.onmessage = (ev) => {
    // Parse het JSON-bericht
    const msg = JSON.parse(ev.data);

    // Alleen berichten van type 'detections' zijn interessant
    if (msg.type !== "detections") return;

    // Haal de lijst met detecties op (items = array van {label, conf})
    const items = msg.items || [];

    // Maak de tabel eerst leeg
    tbody.innerHTML = "";

    // Als er geen objecten gedetecteerd zijn → toon melding
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="2">No objects</td></tr>`;
      return;
    }

    // Voor elk gedetecteerd object: maak een nieuwe rij
    for (const it of items) {
      const tr = document.createElement("tr");   // nieuwe rij
      const tdL = document.createElement("td");  // kolom voor label
      const tdC = document.createElement("td");  // kolom voor confidence

      // Vul kolommen in met label en confidence %
      tdL.textContent = it.label;
      tdC.textContent = `${it.conf}%`;

      // Voeg de cellen toe aan de rij
      tr.appendChild(tdL);
      tr.appendChild(tdC);

      // Voeg de rij toe aan de tabel-body
      tbody.appendChild(tr);
    }
  };
})();
