# Daf-AI

**Daf-AI** is een proof-of-concept applicatie voor real-time AI-objectdetectie met behulp van de camera van een mobiel toestel.  
De video wordt via WebRTC naar een lokale server gestuurd, waar YOLOv8 (Ultralytics) de beelden analyseert. Op je desktop kun je live meekijken en de detecties in real-time volgen.

---

## Functies
- Mobiele camera streaming: gebruik je smartphonecamera als invoer.
- Desktop viewer: bekijk de live videostream inclusief gedetecteerde objecten.
- AI-detectie met YOLOv8: herkent objecten en toont labels + confidence.
- Real-time WebRTC: lage latency, directe feedback.
- Werkt lokaal: geen internetverbinding vereist, enkel lokaal netwerk.

---

## Installatie

### Vereisten
- Windows (PowerShell 7 aanbevolen) of Linux
- [Python 3.11+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

### Stappen
1. **Clone de repository**
   ```bash
   git clone https://github.com/<jouw-username>/Daf-AI.git
   cd Daf-AI
   
2. **Voer het installatiescript uit**
Op Windows (PowerShell):
pwsh ./run.ps1

Dit script:
Maakt automatisch een Python virtual environment (.venv)
Installeert alle vereiste Python-packages
Genereert een lokaal SSL-certificaat (vereist voor mobiel cameragebruik)
Opent poort 8000 in de Windows firewall
Start de server met Uvicorn

Op Linux kan je rechtstreeks de server draaien:
python3 server.py


### Gebruik
Desktop viewer openen
Ga in je browser naar:

https://'lokaal-ip':8000
(Vervang <lokaal-ip> door het IP-adres van je PC in je lokale netwerk, bv. 192.168.1.23.)

Mobiel toestel verbinden
Open in de browser van je smartphone:

https://'lokaal-ip':8000/sender
Sta cameratoegang toe
Klik op “Start stream to server”

### Resultaten bekijken
Op je desktop zie je de videostream met objectdetectie-overlay
In de tabel naast de video worden de labels + confidence-percentages weergegeven

### Tips
Zorg dat zowel je desktop als smartphone op hetzelfde netwerk zitten.
Op iOS en Android kan een zelf-gesigneerd certificaat een waarschuwing geven. Klik dan “Doorgaan” of “Proceed” om verder te gaan.
