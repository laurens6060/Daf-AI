# ==============================
# Realtime AI-detectie server (low-latency)
# - Telefoon (sender.html) stuurt video via WebRTC naar deze server
# - YOLO (Ultralytics) draait in een aparte worker-thread (geen blokkade in WebRTC-pad)
# - /video_feed geeft MJPEG-preview met getekende kaders/labels
# - /ws stuurt live detecties (label + confidence %) naar viewer.html
# - HTTPS wordt gebruikt als key.pem/cert.pem aanwezig zijn (vereist voor camera op mobiel)
# ==============================

import asyncio
import json
import time
import threading
import queue
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole

# ==============================
# Instelbare parameters (tune voor performance/kwaliteit)
# ==============================
IMGSZ = 640            # lange zijde voor inferentie-resize (512 of 640 is doorgaans goed)
CONF  = 0.35           # confidence threshold (hogere waarde = minder valse positieven, sneller NMS)
IOU   = 0.45           # IoU voor NMS
MAX_DET = 100          # max aantal detecties per frame
MJPEG_QUALITY = 70     # JPEG-kwaliteit voor /video_feed (60–75 is goede sweet spot)
VIDEO_FPS_HINT = 30    # alleen cosmetisch voor commentaar; throttle doen we in worker indien gewenst

# ==============================
# FastAPI setup
# ==============================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ==============================
# YOLO-model laden (nano = snel/klein, fijn voor CPU)
# word automatisch gedownload als het model ontbreekt.
# ==============================
model = YOLO("yolov8n.pt")
# Optioneel kleine optimalisatie (doet niets kwaad als het niet helpt):
try:
    model.fuse()  # fuse Conv+BN -> iets snellere inferentie op CPU/GPU
except Exception:
    pass

# ==============================
# Globale staat
# ==============================
latest_frame = None            # laatst geannoteerde BGR frame (voor MJPEG-preview)
latest_detections = []         # [(label:str, conf:float[0..1]), ...]
ws_clients: List[WebSocket] = []  # verbonden WebSocket-klanten (viewer.js)

# Queue waarin we ALLEEN het laatste frame bewaren (maxsize=1 voorkomt buffer-opbouw)
frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)

# Referentie naar de main asyncio-loop om vanuit worker thread WS-berichten te kunnen schedulen
main_loop: asyncio.AbstractEventLoop | None = None

# ==============================
# WebSocket broadcast helper
# ==============================
async def broadcast_ws(obj: dict):
    """
    Stuur JSON naar alle verbonden /ws clients.
    Verwijder clients die gesloten zijn.
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

# ==============================
# Worker-thread: pakt frames uit de queue en draait YOLO
# ==============================
def infer_loop():
    """
    Draait continu in een aparte thread:
    - haalt steeds het nieuwste frame uit frame_q
    - resizet naar IMGSZ (aspect ratio behouden)
    - draait YOLO predict
    - tekent kaders + labels op origineel formaat
    - update latest_frame / latest_detections
    - pusht detecties via WebSocket (via main asyncio loop)
    """
    global latest_frame, latest_detections, main_loop

    while True:
        img = frame_q.get()  # blokkeert tot er een nieuw frame is (laatste, want queue=1)

        # --- Resize voor inferentie (behoud aspect ratio) ---
        H, W = img.shape[:2]
        long_side = max(H, W)
        if long_side != IMGSZ:
            scale = IMGSZ / float(long_side)
            inf_img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            inf_img = img

        # --- YOLO predict (synchroon; we zitten in thread, dus geen asyncio-blokkade) ---
        r = model.predict(
            inf_img, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False, max_det=MAX_DET
        )[0]

        # --- Annotatie tekenen op kopie van ORIGINEEL formaat ---
        annotated = img.copy()
        dets = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0].item() if b.conf is not None else 0.0)
                cls  = int(b.cls[0].item())
                label = model.names.get(cls, str(cls))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, f"{label} {conf*100:.0f}%",
                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                dets.append((label, conf))

        # --- Globale staat updaten (preview + lijst met detecties) ---
        latest_frame = annotated
        latest_detections = dets

        # --- WebSocket notificatie plannen op de main asyncio-loop ---
        if main_loop and main_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    broadcast_ws({
                        "type": "detections",
                        "items": [{"label": l, "conf": round(c * 100, 1)} for (l, c) in dets]
                    }),
                    main_loop
                )
            except Exception:
                # Als schedule faalt (bijv. server shutdown) negeren we dat gewoon
                pass

# Worker één keer starten
_worker = threading.Thread(target=infer_loop, daemon=True)
_worker.start()

# ==============================
# WebRTC: Eigen track die frames opvangt en in de queue stopt
# (zonder zelf detectie te doen — dat doet de worker-thread)
# ==============================
class VideoSinkTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self.track = track

    async def recv(self):
        """
        Low-latency pad:
        - Ontvang frame van WebRTC
        - Stop het als 'laatste frame' in de queue (overschrijft oud frame)
        - Retourneer meteen om de WebRTC-ontvangst niet te blokkeren
        """
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Queue bevat max 1 item: zo vermijden we latency door buffer-opbouw.
        try:
            if frame_q.full():
                frame_q.get_nowait()  # oud frame weggooien
            frame_q.put_nowait(img)  # nieuwste frame plaatsen
        except queue.Full:
            # zou niet mogen gebeuren door de check, maar safety first
            pass

        return frame  # we sturen geen video terug via RTC; preview = /video_feed

# ==============================
# HTTP-pagina's
# ==============================
@app.get("/", response_class=HTMLResponse)
async def viewer(request: Request):
    """Viewerpagina: toont MJPEG + tabel met detecties (via /ws)."""
    return templates.TemplateResponse("viewer.html", {"request": request})

@app.get("/sender", response_class=HTMLResponse)
async def sender(request: Request):
    """Senderpagina: draait op telefoon; vraagt camera en stuurt via WebRTC."""
    return templates.TemplateResponse("sender.html", {"request": request})

# ==============================
# WebSocket voor live detecties (viewer.js)
# ==============================
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        # Client kan heartbeats/pings sturen; we lezen ze om de verbinding open te houden.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)

# ==============================
# MJPEG preview endpoint
# - Stuurt continu JPEG-plaatjes in multipart stream
# - De MJPEG-kwaliteit is instelbaar via MJPEG_QUALITY
# ==============================
@app.get("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = latest_frame
            if frame is None:
                # Zwart placeholder frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                ok, buf = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
            else:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
            if not ok:
                continue
            jpg = buf.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

            # ~33 FPS preview; pas aan indien gewenst
            time.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

# ==============================
# WebRTC signaling (SDP offer/answer)
# - Telefoon post /offer met een SDP-offer
# - Server maakt PeerConnection, voegt VideoSinkTrack toe
# - Server antwoordt met SDP-answer
# ==============================
pcs: List[RTCPeerConnection] = []

@app.post("/offer")
async def offer(sdp: dict):
    pc = RTCPeerConnection()
    pcs.append(pc)
    media_blackhole = MediaBlackhole()  # audio negeren

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            # Vang de videotrack en leid door naar onze low-latency sink
            pc.addTrack(VideoSinkTrack(track))
        else:
            # Audio gebruiken we niet in deze demo
            asyncio.ensure_future(media_blackhole.start())

    @pc.on("connectionstatechange")
    async def on_state_change():
        # Opruimen als de verbinding faalt of sluit
        if pc.connectionState in ("failed", "closed", "disconnected"):
            try:
                await pc.close()
            except Exception:
                pass
            if pc in pcs:
                pcs.remove(pc)

    # SDP-verwerking
    offer = RTCSessionDescription(sdp["sdp"], sdp["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

# ==============================
# FastAPI lifecycle: main asyncio-loop bewaren
# - Zo kan de worker-thread WS-berichten plannen via run_coroutine_threadsafe
# ==============================
@app.on_event("startup")
async def _on_startup():
    global main_loop
    main_loop = asyncio.get_running_loop()

# ==============================
# Entrypoint: start uvicorn
# - HTTPS als key.pem/cert.pem aanwezig (aanrader voor mobiel)
# ==============================
if __name__ == "__main__":
    import uvicorn, os
    ssl_key = "key.pem" if os.path.exists("key.pem") else None
    ssl_crt = "cert.pem" if os.path.exists("cert.pem") else None
    uvicorn.run(
        app, host="0.0.0.0", port=8000,
        ssl_keyfile=ssl_key, ssl_certfile=ssl_crt
    )
