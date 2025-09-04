# ==============================
# Realtime AI-detectie server (tracking + stabilisatie, sliders via /api/config)
# ==============================
# Overzicht:
# - De telefoon (client) stuurt via WebRTC een videostream naar deze server (aiortc).
# - Een worker-thread leest telkens het meest recente frame en runt YOLOv8 tracking (Ultralytics).
# - We gebruiken tracking (ByteTrack) + smoothing (EMA) + debounce (min_hits) + hold (hold_ms)
#   om flikkeren te voorkomen en stabiele boxen/labels te tonen.
# - Voor de desktop viewer leveren we:
#   * Een MJPEG-preview op /video_feed
#   * Een WebSocket /ws voor live detectiegegevens (labels + 'present' aantallen)
#   * Een configuratie-API /api/config om model/thresholds/filters/stabilisaties live te wijzigen.

import os
import asyncio
import json
import time
import threading
import queue
from typing import List, Dict, Optional
from collections import Counter

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from ultralytics import YOLO
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole

# ------------------------------
# Defaults / tunables (basisinstellingen + standaardwaarden)
# ------------------------------
DEFAULT_IMGSZ = 640           # Invoerresolutie voor YOLO (lange zijde in pixels). Ultralytics schaalt intern.
DEFAULT_CONF  = 0.35          # Confidence-threshold (hoger = strenger, minder valse positieven)
DEFAULT_IOU   = 0.45          # IoU-threshold (hoeveel overlap voor NMS/samenvoegen)
MAX_DET       = 200           # Max aantal boxen per frame
MJPEG_QUALITY = 70            # JPEG-kwaliteit voor MJPEG-preview (0..100)

# Stabilisatieparameters (kunnen live aangepast worden via /api/config)
DEFAULT_HOLD_MS   = 500       # Hoe lang (ms) een object zichtbaar blijft na korte 'miss' (anti-flikker)
DEFAULT_MIN_HITS  = 2         # Minimaal aantal opeenvolgende 'hits' voordat we een object tekenen (debounce)
DEFAULT_EMA_ALPHA = 0.4       # Smoothing factor (0..1). Lager = stabieler maar trager reagerend.

TRACKER_CFG = "bytetrack.yaml"  # Ultralytics tracking-config (ByteTrack)

# Beschikbare YOLO-modellen (vriendelijke naam -> pad/bestand)
AVAILABLE_MODELS = {
    "yolov8n": "yolov8n.pt",  # klein/snel
    "yolov8s": "yolov8s.pt",  # iets groter/nauwkeuriger
}

# ------------------------------
# FastAPI setup (static files + templates)
# ------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # JS/CSS etc.
templates = Jinja2Templates(directory="templates")                    # viewer.html / sender.html

# ------------------------------
# Live configuratie-object
# ------------------------------
# Dit Pydantic-model beschrijft alle instelbare parameters. De viewer kan deze via /api/config GET/POST lezen/aanpassen.
class AppConfig(BaseModel):
    model_key: str = Field(default="yolov8n")                      # huidige model-naam (key in AVAILABLE_MODELS)
    imgsz: int = Field(default=DEFAULT_IMGSZ)                      # lange zijde voor het model
    conf: float = Field(default=DEFAULT_CONF, ge=0.0, le=1.0)      # confidence-threshold
    iou:  float = Field(default=DEFAULT_IOU,  ge=0.0, le=1.0)      # IoU-threshold
    allowed_classes: List[str] = Field(default_factory=list)       # lijst met toegestane labels; leeg = alles

    # Stabilisatie (zichtbaar als sliders in de viewer)
    hold_ms: int = Field(default=DEFAULT_HOLD_MS)                  # ms vasthouden bij miss
    min_hits: int = Field(default=DEFAULT_MIN_HITS)                # min. hits voor we tekenen
    ema_alpha: float = Field(default=DEFAULT_EMA_ALPHA, ge=0.0, le=1.0)  # smoothing voor bbox/conf

config = AppConfig()

# ------------------------------
# Model laden/wisselen (thread-safe)
# ------------------------------
_model_lock = threading.Lock()
_model: Optional[YOLO] = None

def load_model(key: str) -> YOLO:
    """
    Laadt een YOLO-model op basis van de 'key' zoals gedefinieerd in AVAILABLE_MODELS.
    """
    path = AVAILABLE_MODELS.get(key)
    if not path:
        raise ValueError(f"Onbekend model: {key}")
    m = YOLO(path)
    # m.fuse() kan kleine snelheidswinst geven door conv+bn samen te voegen (indien ondersteund).
    try:
        m.fuse()
    except Exception:
        pass
    return m

def set_model(key: str):
    """ Wisselt het actieve YOLO-model. """
    global _model
    m = load_model(key)
    with _model_lock:
        _model = m

def get_model() -> YOLO:
    """ Haal het huidige YOLO-model thread-safe op. """
    with _model_lock:
        return _model

def get_model_classes() -> List[str]:
    """
    Retourneert de lijst met klasselabels van het actieve model.
    Ultralytics bewaart namen in m.names (dict of list).
    """
    m = get_model()
    if isinstance(m.names, dict):
        return [str(m.names[i]) for i in sorted(m.names.keys())]
    if isinstance(m.names, list):
        return [str(x) for x in m.names]
    return []

# Init: laad het startmodel
set_model(config.model_key)

# ------------------------------
# Globale staat (laatste frame, laatste detecties, websocket-clients)
# ------------------------------
latest_frame = None                              # geannoteerd BGR-frame voor MJPEG
latest_detections: List[tuple[str, float]] = []  # [(label, conf)] voor de tabel
ws_clients: List[WebSocket] = []                 # actieve WS-clients (viewers)

# Low-latency queue voor frames: we bewaren enkel het meest recente frame
frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
main_loop: asyncio.AbstractEventLoop | None = None  # event loop referentie voor cross-thread WS

# Track-staat per ID (coördinaten op originele resolutie).
# We bewaren per object-id: label, bbox, conf, aantal hits en tijdstip van laatste detectie.
# Dit maakt het mogelijk om labels stabiel te houden en flikkeren te voorkomen.
# Voorbeeld:
#   track_states[42] = {"label": "tv", "bbox": (x1,y1,x2,y2), "conf": 0.79, "hits": 5, "last_ts": 1690000000.0}
track_states: dict[int, dict] = {}

# ------------------------------
# Hulpfuncties (EMA-smoothing, class-filter)
# ------------------------------
def _ema(prev: float, new: float, a: float) -> float:
    """ Exponentiële moving average voor een scalar (bv. confidence). """
    return prev * (1.0 - a) + new * a

def _ema_bbox(prev_bbox, new_bbox, a: float):
    """ EMA voor bounding box-coördinaten (x1,y1,x2,y2). """
    if prev_bbox is None:
        return new_bbox
    return tuple(int(_ema(p, n, a)) for p, n in zip(prev_bbox, new_bbox))

def allowed_filter(label: str) -> bool:
    """
    Filtert op toegestane klassen.
    - Als allowed_classes leeg is: alles is toegestaan.
    - Anders: alleen labels die in allowed_classes staan (case-insensitive).
    """
    if not config.allowed_classes:
        return True
    return label.lower() in {c.lower() for c in config.allowed_classes}

# ------------------------------
# Worker thread: tracking + stabilisatie + WS push
# ------------------------------
def infer_loop():
    """
    Werker die in een aparte thread draait.
    - Leest het meest recente frame uit de queue (low-latency; oudere frames worden gedropt).
    - Voert YOLO.track() uit op het originele frame (GEEN handmatige cv2.resize → geen 'drift').
    - Past EMA-smoothing toe op bbox/conf.
    - Gebruikt debounce (min_hits) en hold (hold_ms) om flikkeren te voorkomen.
    - Bouwt 'present' op: aantallen per label op dit moment in beeld (gebaseerd op actieve track-IDs).
    - Stuurt de resultaten naar alle WS-clients en update het MJPEG-beeld.
    """
    global latest_frame, latest_detections, track_states

    while True:
        # Neem het nieuwste frame; als de queue voller is, droppen we oudere frames (zie VideoSinkTrack.recv).
        img = frame_q.get()
        now = time.time()

        model = get_model()

        # YOLO tracking met persist=True levert stabiele object-IDs over frames.
        # Ultralytics verwerkt intern de schaal (imgsz) en rapporteert boxen in ORIGINELE coördinaten.
        r = model.track(
            img,
            imgsz=config.imgsz, conf=config.conf, iou=config.iou,
            verbose=False, max_det=MAX_DET,
            tracker=TRACKER_CFG, persist=True
        )[0]

        # --- Track-staten bijwerken op basis van huidige detecties ---
        seen_ids = set()
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes
            ids = boxes.id
            for i in range(len(boxes)):
                # Sommige detections kunnen (heel zelden) geen ID krijgen: sla die over.
                if ids is None or ids[i] is None:
                    continue

                obj_id = int(ids[i].item())
                cls = int(boxes.cls[i].item())
                label = model.names.get(cls, str(cls))

                # Toegestane classes filteren
                if not allowed_filter(label):
                    continue

                conf = float(boxes.conf[i].item() if boxes.conf is not None else 0.0)
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                seen_ids.add(obj_id)
                st = track_states.get(obj_id)

                if st is None:
                    # Eerste keer gezien: initieer staat
                    st = {"label": label, "bbox": (x1, y1, x2, y2), "conf": conf, "hits": 1, "last_ts": now}
                else:
                    # Volgende keer: smooth bbox en confidence, update label/ts/hits
                    st["label"] = label
                    st["bbox"]  = _ema_bbox(st["bbox"], (x1, y1, x2, y2), config.ema_alpha)
                    st["conf"]  = _ema(float(st["conf"]), conf, config.ema_alpha)
                    st["hits"]  = int(st["hits"]) + 1
                    st["last_ts"] = now

                track_states[obj_id] = st

        # --- Verwijder 'verouderde' tracks na hold_ms ---
        # Hierdoor blijft een object nog even zichtbaar bij korte misses (anti-flikker).
        expire = []
        for obj_id, st in track_states.items():
            age_ms = (now - float(st["last_ts"])) * 1000.0
            if age_ms > config.hold_ms:
                expire.append(obj_id)
        for obj_id in expire:
            track_states.pop(obj_id, None)

        # --- Teken alleen stabiele/zekere tracks (na min_hits) ---
        annotated = img.copy()
        table_items: List[tuple[str, float]] = []  # voor de tabel (label + confidence)
        active_labels: List[str] = []              # lijst met labels van getoonde tracks (voor 'present')

        for st in track_states.values():
            if st["hits"] < config.min_hits:
                # Nog niet lang genoeg stabiel → (nog) niet tekenen
                continue

            x1, y1, x2, y2 = map(int, st["bbox"])
            label = st["label"]
            conf  = float(st["conf"])

            # Box en label tekenen op het geannoteerde frame
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"{label} {conf*100:.0f}%",
                (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            table_items.append((label, conf))
            active_labels.append(label)

        # Update globale laatste frame + tabelitems
        latest_frame = annotated
        latest_detections = table_items

        # --- 'Present' (in beeld) per label tellen op basis van actieve/zichtbare tracks ---
        present_counts = Counter(active_labels)
        present_list = [{"label": k, "count": int(v)} for k, v in sorted(present_counts.items())]

        # --- Push naar alle WebSocket-clients ---
        # Let op: we zitten in een worker-thread; daarom gebruiken we run_coroutine_threadsafe.
        if main_loop and main_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    broadcast_ws({
                        "type": "detections",
                        "items": [{"label": l, "conf": round(c*100, 1)} for (l, c) in table_items],
                        "present": present_list
                    }),
                    main_loop
                )
            except Exception:
                # WS-fouten hier stilhouden; de volgende iteratie probeert opnieuw
                pass

# Start de achtergrondthread die inferentie en tracking doet
_worker = threading.Thread(target=infer_loop, daemon=True)
_worker.start()

# ------------------------------
# aiortc sink: enkel het nieuwste frame bewaren (low-latency)
# ------------------------------
class VideoSinkTrack(MediaStreamTrack):
    """
    WebRTC 'sink' track die de inkomende videoframes ontvangt van de telefoon.
    We converteren het aiortc-frame naar een BGR numpy-array (OpenCV) en stoppen die in frame_q.
    Door de queue op maxsize=1 te zetten, houden we alleen het meest recente frame bij
    (oude frames worden gedropt → lage latency).
    """
    kind = "video"
    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self.track = track

    async def recv(self):
        # Wacht op volgend frame van de WebRTC-peer
        frame = await self.track.recv()
        # Converteer naar BGR (OpenCV formaat)
        img = frame.to_ndarray(format="bgr24")
        try:
            # Als de queue vol zit, drop het oude frame en plaats de nieuwe
            if frame_q.full():
                frame_q.get_nowait()
            frame_q.put_nowait(img)
        except queue.Full:
            pass
        # We geven het frame terug aan de pipeline (hoewel we het niet verder doorsturen)
        return frame

# ------------------------------
# WebSocket broadcast helper
# ------------------------------
async def broadcast_ws(obj: dict):
    """
    Stuurt een JSON-bericht naar alle verbonden WS-clients.
    Verwijdert clients die niet meer bereikbaar zijn.
    """
    dead = []
    data = json.dumps(obj)
    for ws in ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            ws_clients.remove(ws)
        except ValueError:
            pass

# ------------------------------
# HTTP-pagina's (viewer + sender)
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def viewer(request: Request):
    """
    Viewer-pagina voor de desktop: toont MJPEG-video + tabel + sliders/filters.
    """
    return templates.TemplateResponse("viewer.html", {"request": request})

@app.get("/sender", response_class=HTMLResponse)
async def sender(request: Request):
    """
    Sender-pagina voor de telefoon: vraagt camera-permissies en initieert WebRTC naar /offer.
    """
    return templates.TemplateResponse("sender.html", {"request": request})

# ------------------------------
# WebSocket endpoint voor viewer
# ------------------------------
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """
    WebSocket-verbinding met de viewer:
    - Stuurt bij connect de huidige config, beschikbare modellen en klassen.
    - Daarna ontvangt deze server enkel keepalives/lege berichten (client luistert vooral).
    """
    await websocket.accept()
    ws_clients.append(websocket)
    # Init-bericht: config + modellen + klasselijst van huidig model
    await websocket.send_text(json.dumps({
        "type": "config",
        "config": config.model_dump(),
        "models": list(AVAILABLE_MODELS.keys()),
        "classes": get_model_classes()
    }))
    try:
        while True:
            # We verwachten hier geen specifieke content; dit houdt de WS open.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)

# ------------------------------
# MJPEG preview (/video_feed)
# ------------------------------
@app.get("/video_feed")
def video_feed():
    """
    Levert een MJPEG-stream (multipart/x-mixed-replace) die de viewer gebruikt om live beeld te tonen.
    We encoden telkens het laatste geannoteerde frame als JPEG en sturen dat in een eindeloze boundary-stream.
    """
    def gen():
        while True:
            frame = latest_frame
            if frame is None:
                # Als we nog geen frame hebben: toon een zwart placeholder-beeld
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                ok, buf = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
            else:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])

            if not ok:
                # Encoding mislukte; sla deze iteratie over
                continue

            jpg = buf.tobytes()
            # multipart boundary + content headers + JPEG bytes
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            # Kleine pauze om CPU te sparen (~33 FPS)
            time.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

# ------------------------------
# WebRTC signaling (/offer)
# ------------------------------
pcs: List[RTCPeerConnection] = []

@app.post("/offer")
async def offer(sdp: dict):
    """
    Signaling endpoint voor WebRTC:
    - Ontvangt SDP-offer van de telefoon (sender.html).
    - Zet als remote description, maakt een SDP-answer en retourneert die naar de client.
    - Bij binnenkomende tracks voegen we een VideoSinkTrack toe zodat we frames kunnen ophalen.
    """
    pc = RTCPeerConnection()
    pcs.append(pc)
    blackhole = MediaBlackhole()  # audio negeren

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            # Voor video gebruiken we onze eigen sink die het laatste frame in de queue zet
            pc.addTrack(VideoSinkTrack(track))
        else:
            # Voor audio doen we niets (stop het in het 'zwarte gat')
            asyncio.ensure_future(blackhole.start())

    @pc.on("connectionstatechange")
    async def on_state_change():
        # Opruimen bij afsluiting/verlies
        if pc.connectionState in ("failed", "closed", "disconnected"):
            try:
                await pc.close()
            except Exception:
                pass
            if pc in pcs:
                pcs.remove(pc)

    # Standaard WebRTC offer/answer flow
    offer = RTCSessionDescription(sdp["sdp"], sdp["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

# ------------------------------
# Config API (GET/POST) incl. stabilisatiesliders
# ------------------------------
@app.get("/api/config")
async def get_config():
    """
    Haal huidige configuratie op (inclusief beschikbare modellen en klasselijst).
    Deze info gebruikt de viewer om de UI (sliders/dropdowns/checkboxes) te vullen.
    """
    return {
        "config": config.model_dump(),
        "models": list(AVAILABLE_MODELS.keys()),
        "classes": get_model_classes()
    }

class ConfigUpdate(BaseModel):
    """
    Partiële update van de configuratie. Alle velden zijn optioneel; alleen
    opgegeven velden worden gewijzigd.
    """
    model_key: Optional[str] = None
    imgsz: Optional[int] = None
    conf: Optional[float] = None
    iou: Optional[float] = None
    allowed_classes: Optional[List[str]] = None
    # stabilisatie
    hold_ms: Optional[int] = None
    min_hits: Optional[int] = None
    ema_alpha: Optional[float] = None

@app.post("/api/config")
async def update_config(body: ConfigUpdate):
    """
    Past configuratie-instellingen live toe.
    - Modelwissel (herladen)
    - YOLO-parameters (imgsz/conf/iou)
    - Class-filter (allowed_classes)
    - Stabilisatie (hold_ms/min_hits/ema_alpha)
    Stuurt daarna de nieuwe config naar alle WS-clients.
    """
    changed_model = False

    # Model wisselen
    if body.model_key is not None and body.model_key != config.model_key:
        if body.model_key not in AVAILABLE_MODELS:
            return JSONResponse({"error": "Onbekend model"}, status_code=400)
        set_model(body.model_key)
        config.model_key = body.model_key
        changed_model = True

    # Overige parameters bijwerken
    if body.imgsz is not None: config.imgsz = int(body.imgsz)
    if body.conf  is not None: config.conf  = float(body.conf)
    if body.iou   is not None: config.iou   = float(body.iou)
    if body.allowed_classes is not None:
        # Normaliseer: trim spaties; lege lijst betekent 'alles toestaan'
        config.allowed_classes = [c.strip() for c in body.allowed_classes if c.strip()]

    # Stabilisatie updaten
    if body.hold_ms   is not None: config.hold_ms   = int(body.hold_ms)
    if body.min_hits  is not None: config.min_hits  = int(body.min_hits)
    if body.ema_alpha is not None: config.ema_alpha = float(body.ema_alpha)

    # Broadcast nieuwe config (en bij modelwijziging ook de klasselijst)
    msg = {"type": "config", "config": config.model_dump(), "models": list(AVAILABLE_MODELS.keys())}
    if changed_model:
        msg["classes"] = get_model_classes()

    if ws_clients:
        data = json.dumps(msg)
        for ws in list(ws_clients):
            try:
                await ws.send_text(data)
            except Exception:
                pass

    return {"ok": True, "config": config.model_dump()}

# ------------------------------
# Lifecycle hook (bewaar event loop-referentie)
# ------------------------------
@app.on_event("startup")
async def _on_startup():
    """
    Wordt één keer bij opstart aangeroepen; we bewaren de event loop zodat
    de worker-thread safe coroutines kan inplannen (run_coroutine_threadsafe).
    """
    global main_loop
    main_loop = asyncio.get_running_loop()

# ------------------------------
# Entrypoint (Uvicorn)
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    # Indien aanwezige certificaten (key.pem/cert.pem) gevonden worden, start in HTTPS.
    # Voor mobiel cameragebruik is HTTPS vaak vereist (zeker op iOS/Safari).
    ssl_key = "key.pem" if os.path.exists("key.pem") else None
    ssl_crt = "cert.pem" if os.path.exists("cert.pem") else None
    uvicorn.run(
        app, host="0.0.0.0", port=8000,
        ssl_keyfile=ssl_key, ssl_certfile=ssl_crt
    )
