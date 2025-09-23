# ==============================
# Realtime AI-detectie + Training server (YOLOv8 + tracking/stabilisatie)
# - MJPEG preview    : /video_feed
# - WebSocket live   : /ws
# - Config API       : /api/config  (GET/POST)
# - Detect API       : /api/detect  (POST image)
# - Training API     : /api/train   (POST files, GET status)
# - Downloads        : /api/train/{job_id}/download  |  /api/exports  |  /api/exports/download?path=...
# - Upload endpoint  : /api/trainer/upload-image  (+ /uploads static)
# - Products/Collections JSON-CRUD:
#       Collections:  GET/POST /collections, PUT/DELETE /collections/{cid}
#       Products:     GET/POST /products, PUT/DELETE /products/{pid}
# - Pagina's         : / , /sender , /trainer , /axles , /products-ui
# ==============================

import os
import atexit
import asyncio
import json
import time
import shutil
import tempfile
import threading
import queue
from pathlib import Path
from random import shuffle
from typing import List, Dict, Optional, Union, Any
from collections import Counter
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Body, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from ultralytics import YOLO
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from glob import glob

# ------------------------------
# Defaults / tunables
# ------------------------------
DEFAULT_IMGSZ = 640
DEFAULT_CONF  = 0.35
DEFAULT_IOU   = 0.45
MAX_DET       = 200
MJPEG_QUALITY = 70

DEFAULT_HOLD_MS   = 500
DEFAULT_MIN_HITS  = 2
DEFAULT_EMA_ALPHA = 0.4

TRACKER_CFG = "bytetrack.yaml"
if not Path(TRACKER_CFG).exists():
    print("[WARN] ByteTrack config niet gevonden:", TRACKER_CFG)

AVAILABLE_MODELS: Dict[str, str] = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
}

# ------------------------------
# FastAPI setup
# ------------------------------
app = FastAPI()

# Static mounts
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_ROOT = Path("uploads")
TRAINER_UPLOAD_DIR = UPLOAD_ROOT / "trainer"
TRAINER_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploads")

# ------------------------------
# Live Config
# ------------------------------
class AppConfig(BaseModel):
    model_key: str = Field(default="yolov8n")
    imgsz: int = Field(default=DEFAULT_IMGSZ)
    conf: float = Field(default=DEFAULT_CONF, ge=0.0, le=1.0)
    iou:  float = Field(default=DEFAULT_IOU,  ge=0.0, le=1.0)
    allowed_classes: List[str] = Field(default_factory=list)
    hold_ms: int = Field(default=DEFAULT_HOLD_MS)
    min_hits: int = Field(default=DEFAULT_MIN_HITS)
    ema_alpha: float = Field(default=DEFAULT_EMA_ALPHA, ge=0.0, le=1.0)

config = AppConfig()

# ------------------------------
# Model laden/wisselen (thread-safe)
# ------------------------------

_model_lock = threading.Lock()

# Cache met geladen modellen: key -> YOLO
# Waar 'key' een van:
#   - basismodel-key uit AVAILABLE_MODELS (bv. 'yolov8n')
#   - absoluut/relatief pad naar .pt (bv. 'runs/train7/weights/best.pt')
_models: Dict[str, YOLO] = {}

# Actieve modellen in volgorde (eerste is 'primary' voor tracking)
ACTIVE_MODELS: List[str] = []

def _fuse(m: YOLO) -> YOLO:
    try:
        m.fuse()
    except Exception:
        pass
    return m

def _load_base_model_by_key(key: str) -> YOLO:
    path = AVAILABLE_MODELS.get(key)
    if not path:
        raise ValueError(f"Onbekend basismodel-key: {key}")
    return _fuse(YOLO(path))

def _load_model_by_path(path: str) -> YOLO:
    p = Path(path)
    if not p.exists() or p.suffix.lower() != ".pt":
        raise ValueError(f"Modelbestand niet gevonden of geen .pt: {path}")
    return _fuse(YOLO(str(p)))

def _ensure_loaded(key_or_path: str) -> YOLO:
    """Laad uit cache of vanaf disk; key kan basismodel-key of pad zijn."""
    with _model_lock:
        if key_or_path in _models:
            return _models[key_or_path]
    # buiten lock laden (zwaarder), daarna in cache stoppen
    if key_or_path in AVAILABLE_MODELS:
        m = _load_base_model_by_key(key_or_path)
    else:
        m = _load_model_by_path(key_or_path)
    with _model_lock:
        _models[key_or_path] = m
    return m

def set_active_models(keys_or_paths: List[str]):
    """Stel de actieve modellen lijst in (leeg = fallback naar config.model_key)."""
    global ACTIVE_MODELS
    clean = []
    for it in keys_or_paths:
        s = str(it).strip()
        if not s:
            continue
        # validatie/voorladen (gooit exception bij fout)
        _ensure_loaded(s)
        clean.append(s)
    if not clean:
        # fallback: enkel huidig config.model_key
        clean = [config.model_key]
        _ensure_loaded(config.model_key)
    with _model_lock:
        ACTIVE_MODELS = clean

def get_active_models() -> List[str]:
    with _model_lock:
        return list(ACTIVE_MODELS)

def get_primary_model() -> Optional[YOLO]:
    """Eerste actieve model als primary (voor tracking)."""
    am = get_active_models()
    if not am:
        return None
    k = am[0]
    with _model_lock:
        return _models.get(k)

def get_all_active_model_objs() -> List[YOLO]:
    am = get_active_models()
    with _model_lock:
        return [ _models[k] for k in am if k in _models ]

def get_model_classes() -> List[str]:
    """Classes van het primary model (voor UI)."""
    m = get_primary_model()
    if m is None:
        return []
    names = m.names
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names.keys())]
    if isinstance(names, list):
        return [str(x) for x in names]
    return []

# --- Backwards-compat shims (houdt oude API in leven) ---
def set_model(key: str):
    """Compat: zet één basismodel als actief."""
    set_active_models([key])

def set_model_path(path: str):
    """Compat: zet één .pt pad als actief model."""
    set_active_models([path])

def get_model() -> Optional[YOLO]:
    """Compat: retourneert het primary model (eerste actieve)."""
    return get_primary_model()



# Init: laad startmodellen (enkel huidig basismodel als actief)
set_active_models([config.model_key])

# ------------------------------
# Globale staat voor live video
# ------------------------------
latest_frame = None
latest_detections: List[tuple[str, float]] = []
ws_clients: List[WebSocket] = []

frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
main_loop: asyncio.AbstractEventLoop | None = None

track_states: Dict[int, Dict] = {}  # per track-id: label, bbox, conf, hits, last_ts

# ------------------------------
# Helpers (EMA, filters)
# ------------------------------
def _ema(prev: float, new: float, a: float) -> float:
    return prev * (1.0 - a) + new * a

def _ema_bbox(prev_bbox, new_bbox, a: float):
    if prev_bbox is None:
        return new_bbox
    return tuple(int(_ema(p, n, a)) for p, n in zip(prev_bbox, new_bbox))

def allowed_filter(label: str) -> bool:
    if not config.allowed_classes:
        return True
    return label.lower() in {c.lower() for c in config.allowed_classes}

# --- Model discovery (bestaande getrainde modellen) ---
def _list_trained_models() -> list[dict]:
    items = []

    # runs/**/weights/{best,last}.pt
    for pat in ["runs/**/weights/best.pt", "runs/**/weights/last.pt"]:
        for p in sorted(glob(pat, recursive=True)):
            try:
                st = os.stat(p)
                items.append({
                    "name": Path(p).parts[-3] + " / " + Path(p).name,  # bv. train7 / best.pt
                    "path": p,
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                    "source": "run"
                })
            except Exception:
                pass

    # runs/exported/*.pt
    for p in sorted(glob("runs/exported/*.pt")):
        try:
            st = os.stat(p)
            items.append({
                "name": Path(p).name,
                "path": p,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "source": "exported"
            })
        except Exception:
            pass

    # meest recent boven
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


# ------------------------------
# Worker: tracking + stabilisatie + push
# ------------------------------

def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    denom = area_a + area_b - inter + 1e-6
    return inter / denom

def nms_per_class(dets: List[dict], iou_thr: float) -> List[dict]:
    """dets: [{'xyxy':(x1,y1,x2,y2),'conf':0.73,'label':'earbud', 'src':'primary/extra'}]"""
    out = []
    by_cls: Dict[str, List[dict]] = {}
    for d in dets:
        by_cls.setdefault(d['label'], []).append(d)
    for cls, arr in by_cls.items():
        arr.sort(key=lambda d: d['conf'], reverse=True)
        keep = []
        while arr:
            m = arr.pop(0)
            keep.append(m)
            arr = [d for d in arr if _iou_xyxy(m['xyxy'], d['xyxy']) < iou_thr]
        out.extend(keep)
    return out


def infer_loop():
    global latest_frame, latest_detections, track_states
    while True:
        img = frame_q.get()
        now = time.time()

        active_objs = get_all_active_model_objs()
        if not active_objs:
            latest_frame = img
            continue

        primary = active_objs[0]
        others  = active_objs[1:]
        tracker_cfg = TRACKER_CFG if Path(TRACKER_CFG).exists() else "bytetrack.yaml"

        # 1) PRIMARY: track() -> update tracking state + primaire dets
        primary_dets = []
        try:
            r = primary.track(
                img,
                imgsz=config.imgsz,
                conf=config.conf,
                iou=config.iou,
                verbose=False,
                max_det=MAX_DET,
                tracker=tracker_cfg,
                persist=True
            )[0]
        except Exception as e:
            print(f"[infer_loop] YOLO track() error: {e}")
            r = None

        if r is not None and r.boxes is not None and len(r.boxes):
            boxes = r.boxes
            ids = boxes.id
            names = primary.names if hasattr(primary, 'names') else {}
            for i in range(len(boxes)):
                cls_i = int(boxes.cls[i].item())
                label = (names[cls_i] if isinstance(names, dict) and cls_i in names else str(cls_i))
                if not allowed_filter(label):
                    continue
                conf_i = float(boxes.conf[i].item() if boxes.conf is not None else 0.0)
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                # tracking state (alleen primary)
                if ids is not None and ids[i] is not None:
                    obj_id = int(ids[i].item())
                    st = track_states.get(obj_id)
                    if st is None:
                        st = {"label": label, "bbox": (x1, y1, x2, y2), "conf": conf_i, "hits": 1, "last_ts": now}
                    else:
                        st["label"]   = label
                        st["bbox"]    = _ema_bbox(st["bbox"], (x1, y1, x2, y2), config.ema_alpha)
                        st["conf"]    = _ema(float(st["conf"]), conf_i, config.ema_alpha)
                        st["hits"]    = int(st["hits"]) + 1
                        st["last_ts"] = now
                    track_states[obj_id] = st

                primary_dets.append({"xyxy": (x1, y1, x2, y2), "conf": conf_i, "label": label, "src": "primary"})

        # hold_ms cleanup (tracking blijft ongewijzigd)
        expire = []
        for obj_id, st in list(track_states.items()):
            if (now - float(st["last_ts"])) * 1000.0 > config.hold_ms:
                expire.append(obj_id)
        for obj_id in expire:
            track_states.pop(obj_id, None)

        # 2) OTHERS: predict() en verzamel dets
        extra_dets = []
        for m in others:
            try:
                res = m.predict(img, imgsz=config.imgsz, conf=config.conf, iou=config.iou, verbose=False, max_det=MAX_DET)[0]
            except Exception as e:
                print(f"[infer_loop] extra model predict() error: {e}")
                continue
            if res.boxes is None or len(res.boxes) == 0:
                continue
            names = m.names if hasattr(m, 'names') else {}
            for k in range(len(res.boxes)):
                cls_k = int(res.boxes.cls[k].item())
                label_k = (names[cls_k] if isinstance(names, dict) and cls_k in names else str(cls_k))
                if not allowed_filter(label_k):
                    continue
                conf_k = float(res.boxes.conf[k].item() if res.boxes.conf is not None else 0.0)
                x1, y1, x2, y2 = map(int, res.boxes.xyxy[k].tolist())
                extra_dets.append({"xyxy": (x1, y1, x2, y2), "conf": conf_k, "label": label_k, "src": "extra"})

        # 3) SAMENVOEGEN via per-klasse NMS
        combined = nms_per_class(primary_dets + extra_dets, iou_thr=float(config.iou))

        # 4) TEKENEN
        annotated = img.copy()

        # (a) Teken tracking-boxen (primary, gestabiliseerd met EMA/min_hits)
        table_items: List[tuple[str, float]] = []
        active_labels: List[str] = []

        for st in track_states.values():
            if st["hits"] < config.min_hits:
                continue
            tx1, ty1, tx2, ty2 = map(int, st["bbox"])
            tlabel = st["label"]
            tconf  = float(st["conf"])
            cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{tlabel} {tconf*100:.0f}%", (tx1, max(0, ty1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            table_items.append((tlabel, tconf))
            active_labels.append(tlabel)

        # (b) Dedup combined t.o.v. getrackte boxen en teken rest (oranje)
        tracked_xyxy = [tuple(map(int, st["bbox"])) for st in track_states.values()
                        if st["hits"] >= config.min_hits]

        if tracked_xyxy:
            dedup_combined = [d for d in combined if all(_iou_xyxy(d['xyxy'], txy) < 0.5 for txy in tracked_xyxy)]
        else:
            dedup_combined = combined

        for d in dedup_combined:
            x1, y1, x2, y2 = map(int, d['xyxy'])
            label = d['label']
            conf  = float(d['conf'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(annotated, f"{label} {conf*100:.0f}%", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,180,255), 2)
            table_items.append((label, conf))
            active_labels.append(label)

        # 5) Publiceer resultaten
        latest_frame = annotated
        latest_detections = table_items

        present = Counter(active_labels)
        present_list = [{"label": k, "count": int(v)} for k, v in sorted(present.items())]

        if main_loop and main_loop.is_running():
            try:
                items_payload = [{"label": l, "conf": round(c*100, 1)} for (l, c) in table_items]
                asyncio.run_coroutine_threadsafe(
                    broadcast_ws({
                        "type": "detections",
                        "items": items_payload,
                        "present": present_list
                    }),
                    main_loop
                )
            except Exception:
                pass

_worker = threading.Thread(target=infer_loop, daemon=True)
_worker.start()

# ------------------------------
# WebRTC videoinput sink
# ------------------------------
class VideoSinkTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        try:
            if frame_q.full():
                frame_q.get_nowait()
            frame_q.put_nowait(img)
        except queue.Full:
            pass
        return frame

# ------------------------------
# WebSocket broadcast helper
# ------------------------------
async def broadcast_ws(obj: dict):
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
# Pagina's
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def viewer(request: Request):
    return templates.TemplateResponse("viewer.html", {"request": request})

@app.get("/sender", response_class=HTMLResponse)
async def sender(request: Request):
    return templates.TemplateResponse("sender.html", {"request": request})

@app.get("/trainer", response_class=HTMLResponse)
async def trainer_page(request: Request):
    return templates.TemplateResponse("trainer.html", {"request": request})

@app.get("/axles", response_class=HTMLResponse)
async def axles_page(request: Request):
    return templates.TemplateResponse("axles.html", {"request": request})

@app.get("/axles.html")
async def axles_html_redirect():
    return RedirectResponse(url="/axles", status_code=307)

@app.get("/products-ui", response_class=HTMLResponse)
async def products_ui(request: Request):
    return templates.TemplateResponse("products.html", {"request": request})

# ------------------------------
# WebSocket endpoint viewer
# ------------------------------
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    await websocket.send_text(json.dumps({
        "type": "config",
        "config": config.model_dump(),
        "models": list(AVAILABLE_MODELS.keys()),
        "active_models": get_active_models(),
        "classes": get_model_classes()
    }))

    try:
        while True:
            await websocket.receive_text()  # keepalive
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)

# ------------------------------
# MJPEG preview
# ------------------------------
@app.get("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = latest_frame
            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                ok, buf = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
            else:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
            if not ok:
                continue
            jpg = buf.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

# ------------------------------
# WebRTC signaling
# ------------------------------
pcs: List[RTCPeerConnection] = []

@app.post("/offer")
async def offer(sdp: dict = Body(...)):
    pc = RTCPeerConnection()
    pcs.append(pc)
    blackhole = MediaBlackhole()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            async def consume():
                while True:
                    try:
                        frame = await track.recv()
                    except Exception:
                        break
                    img = frame.to_ndarray(format="bgr24")
                    try:
                        if frame_q.full():
                            frame_q.get_nowait()
                        frame_q.put_nowait(img)
                    except queue.Full:
                        pass
            asyncio.create_task(consume())
        else:
            asyncio.ensure_future(blackhole.start())

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            try:
                await pc.close()
            except Exception:
                pass
            if pc in pcs:
                pcs.remove(pc)

    offer_obj = RTCSessionDescription(sdp["sdp"], sdp["type"])
    await pc.setRemoteDescription(offer_obj)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

# ------------------------------
# Config API
# ------------------------------
class ConfigUpdate(BaseModel):
    model_key: Optional[str] = None
    imgsz: Optional[int] = None
    conf: Optional[float] = None
    iou: Optional[float] = None
    allowed_classes: Optional[List[str]] = None
    hold_ms: Optional[int] = None
    min_hits: Optional[int] = None
    ema_alpha: Optional[float] = None

@app.get("/api/config")
async def get_config():
    return {
        "config": config.model_dump(),
        "models": list(AVAILABLE_MODELS.keys()),
        "active_models": get_active_models(),   # <— voeg toe
        "classes": get_model_classes(),
    }

@app.post("/api/config")
async def update_config(body: ConfigUpdate):
    changed_model = False
    if body.model_key is not None and body.model_key != config.model_key:
        if body.model_key not in AVAILABLE_MODELS:
            return JSONResponse({"error": "Onbekend model"}, status_code=400)
        set_model(body.model_key)
        config.model_key = body.model_key
        changed_model = True

    if body.imgsz is not None: config.imgsz = int(body.imgsz)
    if body.conf  is not None: config.conf  = float(body.conf)
    if body.iou   is not None: config.iou   = float(body.iou)
    if body.allowed_classes is not None:
        config.allowed_classes = [c.strip() for c in body.allowed_classes if c.strip()]
    if body.hold_ms   is not None: config.hold_ms   = int(body.hold_ms)
    if body.min_hits  is not None: config.min_hits  = int(body.min_hits)
    if body.ema_alpha is not None: config.ema_alpha = float(body.ema_alpha)

    msg = {
        "type": "config",
        "config": config.model_dump(),
        "models": list(AVAILABLE_MODELS.keys()),
        "active_models": get_active_models(),
    }
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
# Trainer: upload 1 image (bijv. vanaf phone sender)
# ------------------------------
@app.post("/api/trainer/upload-image")
async def trainer_upload_image(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    if not (file.content_type or "").lower().startswith("image/"):
        return JSONResponse({"error": "Only image uploads allowed"}, status_code=400)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"

    uid = uuid4().hex[:12]
    fname = f"{uid}{ext}"
    fpath = TRAINER_UPLOAD_DIR / fname

    content = await file.read()
    await file.close()
    with fpath.open("wb") as f:
        f.write(content)

    url = f"/uploads/trainer/{fname}"
    display_name = name or (file.filename or fname)

    try:
        await broadcast_ws({
            "type": "trainer_image",
            "name": display_name,
            "url": url,
        })
    except Exception:
        pass

    return {"ok": True, "id": uid, "name": display_name, "url": url}

# ------------------------------
# Detect API
# ------------------------------
@app.post("/api/detect")
async def api_detect(
    file: UploadFile = File(...),
    conf: float = Form(None),
    iou: float = Form(None),
    imgsz: int = Form(None),
):
    try:
        content = await file.read()
    finally:
        await file.close()

    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Kon afbeelding niet decoderen"}, status_code=400)

    c = float(conf if conf is not None else config.conf)
    i = float(iou  if iou  is not None else config.iou)
    s = int(imgsz if imgsz is not None else config.imgsz)

    m = get_model()
    if m is None:
        return JSONResponse({"error": "Model niet geladen"}, status_code=503)

    try:
        res = m.predict(img, conf=c, iou=i, imgsz=s, verbose=False, max_det=MAX_DET)[0]
    except Exception as e:
        return JSONResponse({"error": f"Predict error: {e}"}, status_code=500)

    out = []
    if res.boxes is not None and len(res.boxes):
        boxes = res.boxes
        names = m.names
        for k in range(len(boxes)):
            cls = int(boxes.cls[k].item())
            label = names[cls] if isinstance(names, dict) else str(cls)
            confv = float(boxes.conf[k].item() if boxes.conf is not None else 0.0)
            x1, y1, x2, y2 = map(int, boxes.xyxy[k].tolist())
            out.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": label, "conf": confv
            })

    return {"items": out}

# ------------------------------
# Training jobs
# ------------------------------
TRAIN_JOBS: Dict[str, Dict[str, Any]] = {}  # id -> {'status','log','run_dir','export_path'}

_TEMP_DIRS: List[str] = []
@atexit.register
def _cleanup_tmp():
    for p in _TEMP_DIRS:
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

def _normalize_and_split_dataset(ds_root: Path):
    (ds_root / "images").mkdir(parents=True, exist_ok=True)
    (ds_root / "labels").mkdir(parents=True, exist_ok=True)

    for p in list(ds_root.glob("**/*")):
        if p.is_dir():
            continue
        if p.parent.name in ("images", "labels"):
            continue
        if p.name in ("data.yaml",):
            continue

        suffix = p.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png"):
            target = ds_root / "images" / p.name
        elif suffix == ".txt":
            target = ds_root / "labels" / p.name
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(p), str(target))
        except Exception:
            pass

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ds_root / sub).mkdir(parents=True, exist_ok=True)

    imgs = [p for p in (ds_root / "images").glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    imgs = [p for p in imgs if p.name not in ("train", "val")]
    shuffle(imgs)
    if not imgs:
        raise RuntimeError("Geen images gevonden voor training.")

    cut = max(1, int(0.8 * len(imgs)))
    train_imgs, val_imgs = imgs[:cut], imgs[cut:] or imgs[cut:cut+1]

    def pair_copy(image_paths, split_name):
        for ip in image_paths:
            ip_target = ds_root / "images" / split_name / ip.name
            if str(ip_target) != str(ip):
                shutil.copy2(str(ip), str(ip_target))
            stem = ip.stem
            lbl = ds_root / "labels" / f"{stem}.txt"
            if lbl.exists():
                lbl_target = ds_root / "labels" / split_name / lbl.name
                shutil.copy2(str(lbl), str(lbl_target))

    pair_copy(train_imgs, "train")
    pair_copy(val_imgs,   "val")

def _write_data_yaml(ds_dir: Path, class_names: List[str]):
    yaml = f"""path: {ds_dir}
train: images/train
val: images/val
names: {class_names}
"""
    (ds_dir / 'data.yaml').write_text(yaml, encoding='utf-8')

def _run_train(
    job_id: str, ds_dir: Path, class_name: str,
    export_dir: Optional[str], base_model_key: Optional[str],
    *, epochs: Optional[int]=None, batch: Optional[int]=None,
    imgsz: Optional[int]=None, lr0: Optional[float]=None,
    weight_decay: Optional[float]=None, patience: Optional[int]=None,
    augment: bool=False, resume_from: Optional[str]=None
):
    try:
        TRAIN_JOBS[job_id] = TRAIN_JOBS.get(job_id, {})
        TRAIN_JOBS[job_id].update(status='running', log="Dataset voorbereiden...\n", run_dir='', export_path='')

        _normalize_and_split_dataset(ds_dir)
        _write_data_yaml(ds_dir, [class_name])

        # model kiezen:
        # - als resume_from meegegeven is, gebruiken we dat .pt als start
        # - anders standaard basismodel (yolov8n/s)
        if resume_from:
            base_weights = resume_from
            TRAIN_JOBS[job_id]['log'] += f"Resume vanaf checkpoint: {base_weights}\n"
        else:
            model_key = base_model_key or config.model_key
            base_weights = AVAILABLE_MODELS.get(model_key, "yolov8n.pt")
            TRAIN_JOBS[job_id]['log'] += f"Training met basis {base_weights}\n"

        # model initialiseren
        model = YOLO(base_weights)

        # args opbouwen (alleen meegeven wat gezet is)
        train_kwargs = dict(
            data=str(ds_dir / 'data.yaml'),
            imgsz=int(imgsz or config.imgsz),
            epochs=int(epochs or 20),
            batch=int(batch or 16),
        )
        if lr0 is not None:         train_kwargs["lr0"] = float(lr0)
        if weight_decay is not None: train_kwargs["weight_decay"] = float(weight_decay)
        if patience is not None:     train_kwargs["patience"] = int(patience)
        if augment:                  train_kwargs["augment"] = True

        TRAIN_JOBS[job_id]['log'] += "Train args: " + json.dumps({k: v for k, v in train_kwargs.items()}, ensure_ascii=False) + "\n"

        results = model.train(**train_kwargs)

        run_dir = Path(results.save_dir)
        best = run_dir / "weights" / "best.pt"

        TRAIN_JOBS[job_id].update(status='done', run_dir=str(run_dir))
        TRAIN_JOBS[job_id]['log'] += f"Training klaar. Run dir: {run_dir}\n"

        exported_path = ""
        if best.exists():
            set_model_path(str(best))
            TRAIN_JOBS[job_id]['log'] += f"Nieuw model geladen: {best}\n"
            if export_dir:
                out_dir = Path(export_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"{class_name}_{ts}_best.pt"
                shutil.copy2(best, out_path)
                exported_path = str(out_path)
                TRAIN_JOBS[job_id]['log'] += f"Model geëxporteerd naar: {out_path}\n"
        if exported_path:
            TRAIN_JOBS[job_id]['export_path'] = exported_path

    except Exception as e:
        TRAIN_JOBS[job_id].update(status='error')
        TRAIN_JOBS[job_id]['log'] += f"Fout: {e}\n"
    finally:
        try:
            shutil.rmtree(ds_dir, ignore_errors=True)
        except Exception:
            pass


@app.post("/api/train")
async def api_train(
    files: List[UploadFile] = File(...),
    class_name: str = Form("object"),
    export_dir: Optional[str] = Form(None),
    base_model_key: Optional[str] = Form(None),

    # nieuwe velden (optioneel)
    epochs: Optional[int] = Form(None),
    batch: Optional[int] = Form(None),
    imgsz: Optional[int] = Form(None),
    lr0: Optional[float] = Form(None),
    weight_decay: Optional[float] = Form(None),
    patience: Optional[int] = Form(None),
    augment: Optional[bool] = Form(False),

    # resume vanaf bestaand pad (checkpoint)
    resume_from: Optional[str] = Form(None),
):
    job_id = str(time.time()).replace('.', '')
    TRAIN_JOBS[job_id] = {'status': 'running', 'log': '', 'run_dir': ''}

    tmp = Path(tempfile.mkdtemp(prefix="yolo_ds_"))
    _TEMP_DIRS.append(str(tmp))

    for uf in files:
        suffix = Path(uf.filename).suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png"):
            target = tmp / "images" / Path(uf.filename).name
        elif suffix == ".txt":
            target = tmp / "labels" / Path(uf.filename).name
        else:
            await uf.close()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as f:
            shutil.copyfileobj(uf.file, f)
        await uf.close()

    threading.Thread(
        target=_run_train,
        args=(job_id, tmp, class_name, export_dir, base_model_key),
        kwargs=dict(
            epochs=epochs, batch=batch, imgsz=imgsz, lr0=lr0,
            weight_decay=weight_decay, patience=patience,
            augment=augment, resume_from=resume_from
        ),
        daemon=True
    ).start()

    return {"job_id": job_id, "status": "started"}

@app.get("/api/train/{job_id}")
async def api_train_status(job_id: str):
    data = TRAIN_JOBS.get(job_id, {"status": "unknown"})
    if 'export_path' in data:
        return {
            "status": data.get("status"),
            "log": data.get("log", ""),
            "run_dir": data.get("run_dir", ""),
            "export_path": data.get("export_path"),
        }
    return data

# --- veilige pad-check voor downloads ---
def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

@app.get("/api/train/{job_id}/download")
async def api_train_download(job_id: str):
    job = TRAIN_JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "Unknown job"}, status_code=404)

    export_path = job.get("export_path")
    if not export_path:
        return JSONResponse({"error": "No exported model for this job"}, status_code=404)

    p = Path(export_path)
    if not p.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    base = Path("runs/exported")
    base.mkdir(parents=True, exist_ok=True)
    if not _is_subpath(p, base):
        return JSONResponse({"error": "Forbidden path"}, status_code=403)

    return FileResponse(
        path=str(p),
        media_type="application/octet-stream",
        filename=p.name
    )

@app.get("/api/exports")
async def list_exports(dir: Optional[str] = None):
    root = Path(dir or "runs/exported")
    if not root.exists():
        return []
    items = []
    for f in sorted(root.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
        items.append({
            "name": f.name,
            "path": str(f),
            "size": f.stat().st_size,
            "mtime": int(f.stat().st_mtime),
        })
    return items

@app.get("/api/exports/download")
async def download_export(path: str):
    p = Path(path)
    base = Path("runs/exported")
    if not p.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    if not _is_subpath(p, base):
        return JSONResponse({"error": "Forbidden path"}, status_code=403)
    return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

# ------------------------------
# Collections + Products JSON storage
# ------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PRODUCTS_DB = DATA_DIR / "products.json"
COLLECTIONS_DB = DATA_DIR / "collections.json"

def _load_json(p: Path) -> list[dict]:
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return []

def _save_json(p: Path, items: list[dict]) -> None:
    p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_products() -> list[dict]:
    return _load_json(PRODUCTS_DB)

def _save_products(items: list[dict]) -> None:
    _save_json(PRODUCTS_DB, items)

def _load_collections() -> list[dict]:
    return _load_json(COLLECTIONS_DB)

def _save_collections(items: list[dict]) -> None:
    _save_json(COLLECTIONS_DB, items)

# Models
AllowedVal = Union[str, int, float, bool]

class Collection(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str

class Product(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    collection_id: Optional[str] = None
    properties: dict[str, AllowedVal] = Field(default_factory=dict)

# --- Collections CRUD ---
@app.get("/collections")
async def list_collections():
    return _load_collections()

class CollectionCreate(BaseModel):
    name: str

@app.post("/collections")
async def create_collection(payload: CollectionCreate):
    items = _load_collections()
    c = Collection(**payload.model_dump()).model_dump()
    items.append(c)
    _save_collections(items)
    return c

class CollectionUpdate(BaseModel):
    name: Optional[str] = None

@app.put("/collections/{cid}")
async def update_collection(cid: str, payload: CollectionUpdate):
    items = _load_collections()
    for i, it in enumerate(items):
        if it.get("id") == cid:
            for k, v in payload.model_dump(exclude_none=True).items():
                it[k] = v
            items[i] = it
            _save_collections(items)
            return it
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.delete("/collections/{cid}")
async def delete_collection(cid: str):
    cols = _load_collections()
    if not any(c.get("id")==cid for c in cols):
        return JSONResponse({"error":"Not found"}, status_code=404)
    cols = [c for c in cols if c.get("id")!=cid]
    _save_collections(cols)
    # producten loskoppelen
    prods = _load_products()
    changed = False
    for p in prods:
        if p.get("collection_id")==cid:
            p["collection_id"] = None
            changed = True
    if changed:
        _save_products(prods)
    return {"ok": True, "deleted": cid}

# --- Products CRUD + filter ---
@app.get("/products")
async def list_products(collection_id: Optional[str] = None):
    items = _load_products()
    if collection_id:
        items = [it for it in items if it.get("collection_id")==collection_id]
    return items

class ProductCreate(BaseModel):
    name: str
    collection_id: Optional[str] = None
    properties: dict[str, AllowedVal] = Field(default_factory=dict)

@app.post("/products")
async def create_product(payload: ProductCreate):
    items = _load_products()
    p = Product(**payload.model_dump()).model_dump()
    items.append(p)
    _save_products(items)
    return p

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    collection_id: Optional[str] = None
    properties: Optional[dict[str, AllowedVal]] = None

@app.put("/products/{pid}")
async def update_product(pid: str, payload: ProductUpdate):
    items = _load_products()
    for i, it in enumerate(items):
        if it.get("id") == pid:
            data = payload.model_dump(exclude_none=True)
            if "properties" in data:
                it["properties"] = data["properties"]
                data.pop("properties")
            it.update(data)
            items[i] = it
            _save_products(items)
            return it
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.delete("/products/{pid}")
async def delete_product(pid: str):
    items = _load_products()
    new_items = [it for it in items if it.get("id") != pid]
    if len(new_items) == len(items):
        return JSONResponse({"error": "Not found"}, status_code=404)
    _save_products(new_items)
    return {"ok": True, "deleted": pid}

@app.get("/api/models")
async def api_models():
    return _list_trained_models()

class LoadModelReq(BaseModel):
    path: str

@app.post("/api/model/load")
async def api_model_load(req: LoadModelReq):
    p = Path(req.path)
    if not p.exists() or p.suffix.lower() != ".pt":
        return JSONResponse({"error": "Modelbestand niet gevonden"}, status_code=404)
    try:
        set_model_path(str(p))
    except Exception as e:
        return JSONResponse({"error": f"Kon model niet laden: {e}"}, status_code=500)
    return {"ok": True, "path": str(p), "classes": get_model_classes()}

# ------------------------------
# Active models API
# ------------------------------
class ActiveModelsReq(BaseModel):
    models: List[str] = Field(default_factory=list)  # keys en/of .pt-paden

@app.get("/api/models/active")
async def api_models_active():
    return {"active": get_active_models()}

@app.post("/api/models/active")
async def api_models_active_set(req: ActiveModelsReq):
    try:
        set_active_models(req.models or [])
        # push nieuwe config naar WS clients
        msg = {
            "type": "config",
            "config": config.model_dump(),
            "models": list(AVAILABLE_MODELS.keys()),
            "active_models": get_active_models(),
            "classes": get_model_classes(),
        }
        for ws in list(ws_clients):
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                pass
        return {"ok": True, "active": get_active_models()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ------------------------------
# Lifecycle hook
# ------------------------------
@app.on_event("startup")
async def _on_startup():
    global main_loop
    main_loop = asyncio.get_running_loop()

# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    ssl_key = "key.pem" if os.path.exists("key.pem") else None
    ssl_crt = "cert.pem" if os.path.exists("cert.pem") else None
    uvicorn.run(
        app, host="0.0.0.0", port=8000,
        ssl_keyfile=ssl_key, ssl_certfile=ssl_crt
    )
