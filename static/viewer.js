(function () {
  const tbody = document.getElementById("detBody");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => ws.send("hi");
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type !== "detections") return;

    const items = msg.items || [];
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
  };
})();