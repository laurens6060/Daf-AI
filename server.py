import asyncio, json, time
from typing import List, Tuple
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

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load a small YOLO model (good balance for CPU)
model = YOLO("yolov8n.pt")  # automatically downloads if missing

# Global state (single stream demo)
latest_frame = None           # annotated BGR frame
latest_detections = []        # [(label, conf), ...]
ws_clients: List[WebSocket] = []

class VideoSinkTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self.track = track
        self._last_time = 0

    async def recv(self):
        global latest_frame, latest_detections
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Throttle inference slightly (e.g. 20 FPS max)
        now = time.time()
        if now - self._last_time < 1/20:
            # still forward the previous annotated frame to keep preview fresh
            annotated = latest_frame if latest_frame is not None else img
        else:
            self._last_time = now
            # Run YOLO in thread so we don’t block the event loop
            results = await run_in_threadpool(model.predict, img, verbose=False)
            r = results[0]
            annotated = img.copy()
            dets = []
            if r.boxes is not None and len(r.boxes):
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf = float(b.conf[0].item() if b.conf is not None else 0)
                    cls = int(b.cls[0].item())
                    label = model.names.get(cls, str(cls))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{label} {conf*100:.0f}%",
                                (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    dets.append((label, conf))
            latest_detections = dets
            latest_frame = annotated

            # Push to WS clients
            payload = [{"label": l, "conf": round(c*100, 1)} for (l, c) in dets]
            await broadcast_ws({"type": "detections", "items": payload})

        # Create a new frame to keep the pipeline moving (even though we don’t forward video via RTC)
        return frame

async def broadcast_ws(obj):
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            ws_clients.remove(ws)
        except ValueError:
            pass

# ----- Pages -----
@app.get("/", response_class=HTMLResponse)
async def viewer(request: Request):
    return templates.TemplateResponse("viewer.html", {"request": request})

@app.get("/sender", response_class=HTMLResponse)
async def sender(request: Request):
    return templates.TemplateResponse("sender.html", {"request": request})

# ----- WebSocket for detections -----
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive (client may ping)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)

# ----- MJPEG preview (PC viewer uses this) -----
@app.get("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = latest_frame
            if frame is None:
                # blank placeholder
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                ok, buf = cv2.imencode(".jpg", blank)
            else:
                ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg = buf.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)  # ~33 FPS
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

# ----- WebRTC signaling (phone <-> server) -----
pcs: List[RTCPeerConnection] = []

@app.post("/offer")
async def offer(sdp: dict):
    pc = RTCPeerConnection()
    pcs.append(pc)
    media_blackhole = MediaBlackhole()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(VideoSinkTrack(track))
        else:
            # Ignore audio; pipe to blackhole
            asyncio.ensure_future(media_blackhole.start())

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            try:
                await pc.close()
            except Exception:
                pass
            if pc in pcs:
                pcs.remove(pc)

    offer = RTCSessionDescription(sdp["sdp"], sdp["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

if __name__ == "__main__":
    import uvicorn, os
    ssl_key = "key.pem" if os.path.exists("key.pem") else None
    ssl_crt = "cert.pem" if os.path.exists("cert.pem") else None
    uvicorn.run(
        app, host="0.0.0.0", port=8000,
        ssl_keyfile=ssl_key, ssl_certfile=ssl_crt
    )
