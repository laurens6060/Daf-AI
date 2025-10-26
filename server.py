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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Body, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from ultralytics import YOLO
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from glob import glob
import socket
import webbrowser
import re
import math

import cv2 as cv
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from functools import lru_cache
from typing import Dict, Tuple, List, Optional

# from pyueye import ueye
import sys

# ------------------------------
# # Ip adres zoeken / webbrowser automatisch openen
# ------------------------------
def _get_lan_ip() -> str:
    """Probeer je LAN IP te bepalen (werkt offline en zonder externe calls)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # '10.255.255.255' wordt niet echt gecontacteerd dit triggert OS om de juiste NIC te kiezen
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def _open_browser_soon(url: str, delay: float = 1.5):
    """Open de default browser een fractie na serverstart (race-conditions vermijden)."""
    def _go():
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            pass
    threading.Timer(delay, _go).start()

# ------------------------------
# uEye init
# ------------------------------
# def _ueye_init():
#     global pcImageMemory, ueye_width, ueye_height, nBitsPerPixel, pitch, bytes_per_pixel, nRet
#
#     # uEye globals
#     hCam = ueye.HIDS(0)
#     pcImageMemory = ueye.c_mem_p()
#     MemID = ueye.int()
#     ueye_width = ueye.INT()
#     ueye_height = ueye.INT()
#     nBitsPerPixel = ueye.INT()
#     pitch = ueye.INT()
#     bytes_per_pixel = 0
#
#     # Variables
#     hCam = ueye.HIDS(0)             # 0: first available camera
#     sInfo = ueye.SENSORINFO()
#     cInfo = ueye.CAMINFO()
#     MemID = ueye.int()
#     rectAOI = ueye.IS_RECT()
#     channels = 3                    # 3: channels for color mode(RGB); take 1 channel for monochrome
#     m_nColorMode = ueye.INT()       # Y8/RGB16/RGB24/REG32
#
#     # Starts the driver and establishes the connection to the camera
#     nRet = ueye.is_InitCamera(hCam, None)
#     if nRet != ueye.IS_SUCCESS:
#         print("is_InitCamera ERROR")
#     # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
#     nRet = ueye.is_GetCameraInfo(hCam, cInfo)
#     if nRet != ueye.IS_SUCCESS:
#         print("is_GetCameraInfo ERROR")
#     # You can query additional information about the sensor type used in the camera
#     nRet = ueye.is_GetSensorInfo(hCam, sInfo)
#     if nRet != ueye.IS_SUCCESS:
#         print("is_GetSensorInfo ERROR")
#     nRet = ueye.is_ResetToDefault(hCam)
#     if nRet != ueye.IS_SUCCESS:
#         print("is_ResetToDefault ERROR")
#
#     # Set display mode to DIB
#     nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
#
#     # Set the right color mode
#     if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
#         # setup the color depth to the current windows setting
#         ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
#         bytes_per_pixel = int(nBitsPerPixel / 8)
#         print("IS_COLORMODE_BAYER: ", )
#
#     elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
#         # for color camera models use RGB32 mode
#         m_nColorMode = ueye.IS_CM_BGRA8_PACKED
#         nBitsPerPixel = ueye.INT(32)
#         bytes_per_pixel = int(nBitsPerPixel / 8)
#         print("IS_COLORMODE_CBYCRY: ", )
#
#     elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
#         # for color camera models use RGB32 mode
#         m_nColorMode = ueye.IS_CM_MONO8
#         nBitsPerPixel = ueye.INT(8)
#         bytes_per_pixel = int(nBitsPerPixel / 8)
#         print("IS_COLORMODE_MONOCHROME: ", )
#
#     else:
#         # for monochrome camera models use Y8 mode
#         m_nColorMode = ueye.IS_CM_MONO8
#         nBitsPerPixel = ueye.INT(8)
#         bytes_per_pixel = int(nBitsPerPixel / 8)
#         print("else")
#
#     print("- m_nColorMode: \t", m_nColorMode)
#     print("- nBitsPerPixel: \t", nBitsPerPixel)
#     print("- bytes_per_pixel: \t", bytes_per_pixel)
#     print()
#
#     # Can be used to set the size and position of an "area of interest"(AOI) within an image
#     nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
#     if nRet != ueye.IS_SUCCESS:
#         print("is_AOI ERROR")
#
#     ueye_width = rectAOI.s32Width
#     ueye_height = rectAOI.s32Height
#
#     # Prints out some information about the camera and the sensor
#     print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
#     print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
#     print("Maximum image width:\t", ueye_width)
#     print("Maximum image height:\t", ueye_height)
#     print()
#     # TODO Downscale
#     # ueye_width = ueye.c_int(int(ueye_width/4))
#     # ueye_height = ueye.c_int(int(ueye_height/4))
#     print(type(ueye_width))
#     print(type(ueye_height))
#
#     # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
#     nRet = ueye.is_AllocImageMem(hCam, ueye_width, ueye_height, nBitsPerPixel, pcImageMemory, MemID)
#     if nRet != ueye.IS_SUCCESS:
#         print("is_AllocImageMem ERROR")
#     else:
#         # Makes the specified image memory the active memory
#         nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
#         if nRet != ueye.IS_SUCCESS:
#             print("is_SetImageMem ERROR")
#         else:
#             # Set the desired color mode
#             nRet = ueye.is_SetColorMode(hCam, m_nColorMode)
#
#     # Activates the camera's live video mode (free run mode)
#     nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
#     if nRet != ueye.IS_SUCCESS:
#         print("is_CaptureVideo ERROR")
#
#     # Enables the queue mode for existing image memory sequences
#     nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, ueye_width, ueye_height, nBitsPerPixel, pitch)
#     if nRet != ueye.IS_SUCCESS:
#         print("is_InquireImageMem ERROR")
#     else:
#         print("ueye init complete")

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

_last_auto_ref_match_ts = 0.0
AUTO_REF_MATCH_INTERVAL = 0.5  # seconden; pas aan naar smaak

CENTER_BOX_FRAC_W = 0.40   # centrale venster: 40% van beeldbreedte
CENTER_BOX_FRAC_H = 0.40   # centrale venster: 40% van beeldhoogte
CENTER_MIN_FRAMES = 8      # minimaal aantal opeenvolgende frames "in het midden" voor goed-/afkeuren
BORDER_MARGIN_PX  = 8      # cirkel mag niet (bijna) uit beeld steken

_center_gate_state = {
    "last_pid": None,      # laatst gedetecteerde product-id (auto type)
    "stable": 0,           # aaneengesloten frames dat het centrum OK is
}
def _is_centered(cx: float, cy: float, r: float, w: int, h: int) -> bool:
    """Ligt de naaf binnen het centrale venster én niet tegen de rand?"""
    cw, ch = int(w * CENTER_BOX_FRAC_W), int(h * CENTER_BOX_FRAC_H)
    x1 = (w - cw) // 2
    y1 = (h - ch) // 2
    x2 = x1 + cw
    y2 = y1 + ch

    inside_center_box = (x1 <= cx <= x2) and (y1 <= cy <= y2)
    safe_in_frame = (
        (cx - r) >= BORDER_MARGIN_PX and
        (cy - r) >= BORDER_MARGIN_PX and
        (cx + r) <= (w - BORDER_MARGIN_PX) and
        (cy + r) <= (h - BORDER_MARGIN_PX)
    )
    return inside_center_box and safe_in_frame

TRACKER_CFG = "bytetrack.yaml"
if not Path(TRACKER_CFG).exists():
    print("[WARN] ByteTrack config niet gevonden:", TRACKER_CFG)

AVAILABLE_MODELS: Dict[str, str] = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8n-seg": "yolov8n-seg.pt",
    "yolo11n-seg": "yolo11n-seg.pt",
    "yolov8s-seg": "yolov8s-seg.pt",
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
    show_masks: bool = True
    show_boxes: bool = True
    contour_match_enabled: bool = True
    contour_match_class: str = "hole"
    contour_match_iop: float = 0.60
    active_contour_ids: Optional[List[str]] = None
    active_contour_id: Optional[str] = None

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
latest_raw_frame = None
latest_detections: List[tuple[str, float]] = []
ws_clients: List[WebSocket] = []

frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
main_loop: asyncio.AbstractEventLoop | None = None

_infer_thread: Optional[threading.Thread] = None
_cv2_thread: Optional[threading.Thread] = None

track_states: Dict[int, Dict] = {}  # per track-id: label, bbox, conf, hits, last_ts

# ------------------------------
# Globale staat voor ueye camera
# ------------------------------
pcImageMemory = None
ueye_width = 640
ueye_height = 480
nBitsPerPixel = 24
pitch = 0
bytes_per_pixel = 0
nRet = None

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
    #for pat in ["runs/**/weights/best.pt", "runs/**/weights/last.pt"]:
    #    for p in sorted(glob(pat, recursive=True)):
    #       try:
    #           st = os.stat(p)
    #           items.append({
    #               "name": Path(p).parts[-3] + " / " + Path(p).name,  # bv. train7 / best.pt
    #               "path": p,
    #              "size": st.st_size,
    #              "mtime": int(st.st_mtime),
    #              "source": "run"
    #         })
    #      except Exception:
    #          pass

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

        # runs/segment/**/weights/best.pt
        #for pseg in sorted(glob("runs/segment/**/weights/best.pt")):
            #    try:
            #   st = os.stat(pseg)
            #   items.append({
            #       "name": Path(pseg).name,
            #       "path": pseg,
            #       "size": st.st_size,
            #       "mtime": int(st.st_mtime),
            #       "source": "segment"
            #   })
            #except Exception:
        #   pass
    # meest recent boven
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items

def _present_to_map(present_list: list[dict]) -> dict[str, int]:
    # [{"label": "earbud", "count": 2}, ...] -> {"earbud": 2, ...}
    m = {}
    for it in present_list:
        try:
            m[str(it["label"]).strip().lower()] = int(it["count"])
        except Exception:
            pass
    return m

def _get_actual_for(name: str, m: dict[str, int]) -> int:
    # simpele toleranties: enkelvoud/meervoud
    n = str(name).strip().lower()
    cand = {n, (n[:-1] if n.endswith("s") else n + "s")}
    for k in m.keys():
        if k in cand:
            return int(m[k])
    return int(m.get(n, 0))

def _match_products_by_properties(present_list: list[dict]) -> list[str]:
    """Product 'matcht' wanneer ALLE properties >0 exact gehaald zijn."""
    pmap = _present_to_map(present_list)
    out = []
    for p in _load_products():
        props = p.get("properties") or {}
        expected_pairs = [(k, v) for k, v in props.items()
                          if isinstance(v, (int, float)) and int(v) != 0]
        if not expected_pairs:
            continue
        ok = True
        for name, exp in expected_pairs:
            act = _get_actual_for(str(name), pmap)
            if int(act) != int(exp):
                ok = False
                break
        if ok:
            out.append(p.get("id"))
    return out


def _short_base_key(path_or_name: str) -> str:
    """
    Haal 'yolov8n' of 'yolov8n-seg' uit een bestandsnaam/pad.
    Valt terug op 'yolov8n' als er niets matcht.
    """
    m = re.search(r'(yolov8[nsmlx](?:-seg)?)', os.path.basename(str(path_or_name)))
    return m.group(1) if m else 'yolov8n'

def _safe_out_path(out_dir: Path, filename: str) -> Path:
    """
    Voorkom overschrijven: voeg (2), (3), ... toe als de naam al bestaat.
    """
    p = out_dir / filename
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    i = 2
    while (out_dir / f"{stem}({i}){suf}").exists():
        i += 1
    return out_dir / f"{stem}({i}){suf}"
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

def _color_for_label(label: str) -> tuple[int,int,int]:
    """Deterministische BGR kleur per label."""
    h = abs(hash(label)) % 255
    return ( (h*3) % 255, (h*5) % 255, (h*7) % 255 )

def draw_mask_polygons(annotated: np.ndarray, polys: list[np.ndarray], color_bgr: tuple[int,int,int], alpha: float=0.35):
    """
    polys: lijst van Nx2 float arrays (xy), zoals result.masks.xy[i]
    Tekent halftransparante polygon-fill + dunne rand.
    """
    if not polys:
        return
    overlay = annotated.copy()
    for poly in polys:
        if poly is None or len(poly) < 3:  # min. drie punten
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.polylines(overlay, [pts], isClosed=True, color=color_bgr, thickness=2)
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, dst=annotated)

def cv2_loop():
    print("start cv2_loop")
    global latest_frame, latest_detections, track_states
    global pcImageMemory, ueye_width, ueye_height, nBitsPerPixel, pitch, bytes_per_pixel

    # _ueye_init()

    # cache plain ints
    w = int(ueye_width.value if hasattr(ueye_width, "value") else ueye_width)
    h = int(ueye_height.value if hasattr(ueye_height, "value") else ueye_height)
    bpp = int(nBitsPerPixel.value if hasattr(nBitsPerPixel, "value") else nBitsPerPixel)
    pitch_val = int(pitch.value if hasattr(pitch, "value") else pitch)
    ch = int(bpp // 8)

    # while True:
    #     try:
    #         array = ueye.get_data(pcImageMemory, w, h, bpp, pitch_val, copy=False)
    #     except Exception as e:
    #         print("[uEye] get_data failed:", e)
    #         time.sleep(0.05)
    #         continue
    #
    #     try:
    #         frame = np.reshape(array, (h, w, ch)) if ch >= 1 else np.reshape(array, (h, w))
    #     except Exception as e:
    #         print("[uEye] reshape failed:", e)
    #         time.sleep(0.01)
    #         continue
    #
    #     # normalize to 3-channel BGR
    #     if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
    #         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    #     elif frame.ndim == 3 and frame.shape[2] == 4:
    #         frame = frame[:, :, :3]
    #
    #     # optional scale-down
    #     frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #
    #     try:
    #         if frame_q.full():
    #             frame_q.get_nowait()
    #         frame_q.put_nowait(frame)
    #     except queue.Full:
    #         pass


_worker = threading.Thread(target=cv2_loop, daemon=True)
_worker.start()

def infer_loop():
    global latest_frame, latest_detections, track_states, latest_raw_frame

    while True:
        img = frame_q.get()
        latest_raw_frame = img.copy()
        now = time.time()
        update_last_frame(img)
        pid = None
        centered_ok = False
        stable_now = 0
        match_flag = False

        active_objs = get_all_active_model_objs()
        if not active_objs:
            latest_frame = img
            latest_detections = []
            # still broadcast empties if you want; for now just continue
            continue

        primary = active_objs[0]
        others  = active_objs[1:]
        tracker_cfg = TRACKER_CFG if Path(TRACKER_CFG).exists() else "bytetrack.yaml"

        # 1) PRIMARY: track() -> tracking-state & primary dets
        primary_dets = []
        primary_polys = []  # [(label, [poly_xy])]
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
            names = getattr(primary, "names", {})
            for i in range(len(boxes)):
                cls_i = int(boxes.cls[i].item())
                label = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else str(cls_i)
                if not allowed_filter(label):
                    continue
                conf_i = float(boxes.conf[i].item() if boxes.conf is not None else 0.0)
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                # update tracking state (primary only)
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

        # primary masks (if seg)
        if r is not None and getattr(r, "masks", None) is not None and r.masks is not None:
            try:
                masks_xy = r.masks.xy  # List[np.ndarray Nx2]
            except Exception:
                masks_xy = None
            if masks_xy and r.boxes is not None:
                names = getattr(primary, "names", {})
                for i, poly in enumerate(masks_xy):
                    cls_i = int(r.boxes.cls[i].item())
                    label = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else str(cls_i)
                    if not allowed_filter(label):
                        continue
                    if poly is not None and len(poly) >= 3:
                        primary_polys.append((label, [poly]))

        # hold_ms expiry for tracks
        expire = []
        for obj_id, st in list(track_states.items()):
            if (now - float(st["last_ts"])) * 1000.0 > config.hold_ms:
                expire.append(obj_id)
        for obj_id in expire:
            track_states.pop(obj_id, None)

        # 2) OTHERS: predict()
        extra_dets = []
        extra_polys = []
        for m in others:
            try:
                res = m.predict(
                    img, imgsz=config.imgsz, conf=config.conf, iou=config.iou,
                    verbose=False, max_det=MAX_DET
                )[0]
            except Exception as e:
                print(f"[infer_loop] extra model predict() error: {e}")
                continue

            if res.boxes is None or len(res.boxes) == 0:
                continue

            names = getattr(m, "names", {})

            # boxes
            for k in range(len(res.boxes)):
                cls_k = int(res.boxes.cls[k].item())
                label_k = names.get(cls_k, str(cls_k)) if isinstance(names, dict) else str(cls_k)
                if not allowed_filter(label_k):
                    continue
                conf_k = float(res.boxes.conf[k].item() if res.boxes.conf is not None else 0.0)
                x1, y1, x2, y2 = map(int, res.boxes.xyxy[k].tolist())
                extra_dets.append({"xyxy": (x1, y1, x2, y2), "conf": conf_k, "label": label_k, "src": "extra"})

            # masks (seg)
            if getattr(res, "masks", None) is not None and res.masks is not None:
                try:
                    masks_xy = res.masks.xy
                except Exception:
                    masks_xy = None
                if masks_xy:
                    for k, poly in enumerate(masks_xy):
                        cls_k = int(res.boxes.cls[k].item()) if res.boxes is not None else None
                        label_k = names.get(cls_k, str(cls_k)) if isinstance(names, dict) else str(cls_k)
                        if not allowed_filter(label_k):
                            continue
                        if poly is not None and len(poly) >= 3:
                            extra_polys.append((label_k, [poly]))

        # 3) merge via per-class NMS
        combined = nms_per_class(primary_dets + extra_dets, iou_thr=float(config.iou))

        # 4) draw
        # 4) draw
        annotated = img.copy()
        table_items: List[tuple[str, float]] = []
        active_labels: List[str] = []

        # i.p.v. direct tekenen: verzamel draw-opdrachten
        box_draw_cmds = []  # tuples: (kind, (x1,y1,x2,y2), label, conf)

        # (a) tracked boxes (stabilized)
        for st in track_states.values():
            if st["hits"] < config.min_hits:
                continue
            tx1, ty1, tx2, ty2 = map(int, st["bbox"])
            tlabel = st["label"]
            tconf = float(st["conf"])
            box_draw_cmds.append(("tracked", (tx1, ty1, tx2, ty2), tlabel, tconf))
            table_items.append((tlabel, tconf))
            active_labels.append(tlabel)

        # (b) dedup combined vs tracked
        tracked_xyxy = [tuple(map(int, st["bbox"])) for st in track_states.values()
                        if st["hits"] >= config.min_hits]
        dedup_combined = [d for d in combined if all(_iou_xyxy(d['xyxy'], txy) < 0.5 for txy in tracked_xyxy)] \
            if tracked_xyxy else combined

        for d in dedup_combined:
            x1, y1, x2, y2 = map(int, d['xyxy'])
            label = d['label'];
            conf = float(d['conf'])
            box_draw_cmds.append(("combined", (x1, y1, x2, y2), label, conf))
            table_items.append((label, conf))
            active_labels.append(label)

        # (c) masks
        if getattr(config, "show_masks", True):
            for label, polys in primary_polys:
                draw_mask_polygons(annotated, polys, _color_for_label(label), alpha=0.35)
            for label, polys in extra_polys:
                draw_mask_polygons(annotated, polys, _color_for_label(label), alpha=0.25)

        # (d) contour matching (IoP)
        contour_hits = []
        matched_product_ids = set()
        match_regions = []
        templates = _load_contours()
        poi_markers = []  # [{x:int,y:int, ok:bool, label:str, required:int, found:int}]
        poi_fail_mismatches = []  # tekstregels voor bestaande reject-flow

        try:
            if not getattr(config, "contour_match_enabled", True):
                templates = []
            else:
                ids = getattr(config, "active_contour_ids", None)
                if ids is not None:
                    if len(ids) == 0:
                        templates = []
                    else:
                        idset = {str(x) for x in ids}
                        templates = [t for t in templates if str(t.get("id")) in idset]

            if templates:
                h, w = annotated.shape[:2]
                thr = float(getattr(config, "contour_match_iop", 0.60))
                for t in templates:
                    tpl_xy = _denorm_poly(t.get("polygon01", []), w, h)
                    if len(tpl_xy) >= 3:
                        cv2.polylines(annotated, [np.asarray(tpl_xy, np.int32)], True, (255, 255, 0), 2)

                hole_polys = [p for _, polys in (primary_polys + extra_polys) for p in polys if p is not None and len(p) >= 3]

                for t in templates:
                    tpl_xy = _denorm_poly(t.get("polygon01", []), w, h)
                    if len(tpl_xy) < 3:
                        continue
                    best = 0.0
                    for hp in hole_polys:
                        iop = _poly_iop(tpl_xy, hp, w, h)
                        if iop > best:
                            best = iop
                    if best >= thr:
                        contour_hits.append({
                            "type_key": t.get("type_key"),
                            "iop": round(best, 3),
                            **({"product_id": t.get("product_id")} if t.get("product_id") else {})
                        })
                        if t.get("product_id"):
                            matched_product_ids.add(t.get("product_id"))

                        cv2.polylines(annotated, [np.asarray(tpl_xy, np.int32)], True, (0, 255, 0), 3)
                        cv2.putText(annotated, f"match {t.get('type_key')} ({best * 100:.0f}%)",
                                    tuple(tpl_xy[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        rx, ry, rw, rh = cv2.boundingRect(np.asarray(tpl_xy, np.int32))
                        match_regions.append((rx, ry, rx + rw, ry + rh))
        except Exception as e:
            print(f"[contours] match error: {e}")

        # (d2) AUTOMATISCHE TYPE-HERKENNING m.b.v. ref_contours (T8/T9/T13) -> product match
        try:
            global _last_auto_ref_match_ts, _center_gate_state
            now = time.time()
            if (now - _last_auto_ref_match_ts) >= AUTO_REF_MATCH_INTERVAL:
                _last_auto_ref_match_ts = now

                gray_live = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # live contour + signatures + centrum/straal
                cnt_live = _find_outer_contour_detailed(gray_live, debug_name=None)
                (cx, cy, r), live_sigs = _live_features(gray_live, cnt_live)

                # doorloop beste kandidaat per type (zoals in /api/contour-match)
                type_keys = _list_ref_type_keys()
                best = None
                second = None

                for tkey in type_keys:
                    cands = _iter_ref_files(tkey)
                    best_for_type = None
                    for ref_path in cands:
                        try:
                            ref_feats = _load_ref_features(str(ref_path))
                        except Exception:
                            continue

                        cnt_ref = ref_feats["cnt"]
                        score, ncc, s_shape = _combined_score(cnt_ref, cnt_live, ref_feats, live_sigs)
                        ref_mask, live_mask = _align_for_overlap(cnt_ref, cnt_live, out_size=768)
                        iou = _compute_overlap(ref_mask, live_mask)

                        rec = {
                            "type_key": tkey,
                            "ref_file": ref_path.name,
                            "score": float(score),
                            "ncc": float(ncc),
                            "shape": float(s_shape),
                            "iou": float(iou),
                        }
                        if (best_for_type is None) or (rec["score"] < best_for_type["score"]):
                            best_for_type = rec

                    if best_for_type:
                        if (best is None) or (best_for_type["score"] < best["score"]):
                            second = best
                            best = best_for_type
                        elif (second is None) or (best_for_type["score"] < second["score"]):
                            second = best_for_type

                # drempels (gelijk aan endpoint-logica, evt. tunen)
                if best is not None:
                    SCORE_MAX = 0.02  # zelfde default als endpoint-Form
                    NCC_MIN = 0.70
                    IOU_MIN = 0.60
                    MARGIN = 0.18

                    ok_by_thresh = (best["score"] <= SCORE_MAX) and (best["ncc"] >= NCC_MIN) and (
                                best["iou"] >= IOU_MIN)
                    ok_by_iou = best.get("iou", 0.0) >= IOU_MIN
                    ok_by_margin = True
                    if second:
                        rel = (second["score"] - best["score"]) / max(second["score"], 1e-6)
                        ok_by_margin = (rel >= MARGIN)

                    match_flag = bool(
                        (ok_by_thresh)
                        or (best.get("iou", 0.0) >= 0.90 and best["score"] <= 0.40)
                        or (ok_by_margin and best.get("ncc", 0.0) >= 0.65)
                    )

                    # ===== Nieuw: center-gate (alleen bij succesvolle auto-type match) =====
                    H, W = annotated.shape[:2]
                    pid = None
                    centered_ok = False
                    stable_now = 0

                    if match_flag:
                        # map type -> product-id op NAAM (met fallback)
                        pid = _product_id_by_type_key(best["type_key"], best.get("ref_file"))

                        # check centreren
                        centered_ok = _is_centered(float(cx), float(cy), float(r), W, H)

                        # state per product vasthouden
                        if pid != _center_gate_state.get("last_pid"):
                            _center_gate_state["last_pid"] = pid
                            _center_gate_state["stable"] = 0

                        if centered_ok and pid:
                            _center_gate_state["stable"] = int(_center_gate_state.get("stable", 0)) + 1
                        else:
                            # kies hier zelf: reset hard, of laat langzaam afbouwen
                            _center_gate_state["stable"] = 0

                        stable_now = _center_gate_state["stable"]

                        # 1) informatieve rij (zoals contour_hits uit handmatige templates)
                        contour_hits.append({
                            "type_key": best["type_key"],
                            "iop": round(best.get("iou", 0.0), 3),
                            "ref": best.get("ref_file", ""),
                            "centered": bool(centered_ok),
                            "stable": int(stable_now),
                            "need": int(CENTER_MIN_FRAMES),
                        })

                        # 2) markeer pas als 'gematcht' (en dus klaar voor keuring)
                        #    wanneer genoeg frames stabiel in het midden
                        if pid and stable_now >= CENTER_MIN_FRAMES:
                            matched_product_ids.add(pid)

                        # 3) visuele hints (rechthoek + status)
                        #    teken het centrale venster
                        cw, ch = int(W * CENTER_BOX_FRAC_W), int(H * CENTER_BOX_FRAC_H)
                        x1 = (W - cw) // 2;
                        y1 = (H - ch) // 2
                        x2 = x1 + cw;
                        y2 = y1 + ch
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 160, 255), 2)

                        color = (0, 220, 0) if (pid and stable_now >= CENTER_MIN_FRAMES) else (
                            (0, 200, 255) if centered_ok else (0, 180, 255))
                        msg = f"type {best['type_key']} (auto)  "
                        msg += f"[center {'OK' if centered_ok else '…'}: {stable_now}/{CENTER_MIN_FRAMES}]"
                        if pid and stable_now >= CENTER_MIN_FRAMES:
                            msg += "  — READY"
                        cv2.putText(
                            annotated, msg, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                        )

        except Exception as e:
            print("[auto ref type] error:", e)

        # (e) teken nu de boxen; onderdruk tekst wanneer overlapt met match-regio
        if config.show_boxes:
            for kind, (x1, y1, x2, y2), label, conf in box_draw_cmds:
                color = (0, 255, 0) if kind == "tracked" else (0, 180, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                # onderdruk als IoU met eender welke match-regio >= 0.30
                suppress = any(_iou_xyxy((x1, y1, x2, y2), mr) >= 0.30 for mr in match_regions)
                if not suppress:
                    cv2.putText(annotated, f"{label} {conf * 100:.0f}%",
                                (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 6) compute present
        present = Counter(active_labels)
        present_list = [{"label": k, "count": int(v)} for k, v in sorted(present.items())]

        # 6b) fallback: als contouren uit staan of geen templates/matches, probeer property-match
        try:
            contour_enabled = getattr(config, "contour_match_enabled", True)
            # NB: 'templates' bestaat hier nog uit sectie (d)
            needs_fallback = (not contour_enabled) or not templates
            if needs_fallback and present_list:
                fallback_ids = _match_products_by_properties(present_list)
                matched_product_ids.update(fallback_ids)
        except Exception as e:
            print(f"[fallback-match] error: {e}")

            # ---- POI ALTIJD DOEN (niet in except) ----
        def _inside_any_match(x, y, regions):
            for (x1, y1, x2, y2) in regions:
                if x1 <= x <= x2 and y1 <= y <= y2:
                 return True
            return False

        try:
            if matched_product_ids:
                    h, w = annotated.shape[:2]
                    all_pois = _load_pois()
                    active_tpl_ids = {t.get("id") for t in templates} if templates else set()
                    for pid in matched_product_ids:
                        poi_sets = [
                            p for p in all_pois
                            if p.get("product_id") == pid and (
                                    not p.get("contour_id") or p.get("contour_id") in active_tpl_ids)
                        ]
                        for ps in poi_sets:
                            for it in ps.get("items", []):
                                x = int(round(float(it.get("x01", 0)) * w))
                                y = int(round(float(it.get("y01", 0)) * h))
                                r = int(round(float(it.get("radius01", 0.02)) * min(w, h)))
                                expected = str(it.get("expected_label", "")).strip().lower()
                                required = int(it.get("required", 1))

                                if not _inside_any_match(x, y, match_regions):
                                    poi_markers.append({"x": x, "y": y, "r": r, "ok": False,
                                                        "label": expected, "required": required, "found": 0})
                                    poi_fail_mismatches.append(
                                        f"POI buiten contour: verwacht {required}× {expected} bij ({x},{y})")
                                    continue

                                found = 0
                                for kind, (x1, y1, x2, y2), label, conf in box_draw_cmds:
                                    if label.lower() != expected:
                                        continue
                                    cx = (x1 + x2) // 2
                                    cy = (y1 + y2) // 2
                                    if math.hypot(cx - x, cy - y) <= r:
                                        found += 1

                                ok = found >= required
                                poi_markers.append({"x": x, "y": y, "r": r, "ok": ok,
                                                    "label": expected, "required": required, "found": found})
                                if not ok:
                                    poi_fail_mismatches.append(
                                        f"POI mist: {found}/{required}× {expected} bij ({x},{y})")
        except Exception as e:
            print("[poi] error:", e)

        h, w = annotated.shape[:2]
        latest_frame = annotated
        latest_detections = table_items

        if main_loop and main_loop.is_running():
            try:
                items_payload = [{"label": l, "conf": round(c * 100, 1)} for (l, c) in table_items]
                payload = {
                    "type": "detections",
                    "items": items_payload,
                    "present": present_list,
                    "frame_w": w,
                    "frame_h": h,
                    "pid": pid,
                    "centered": bool(centered_ok),
                    "stable": int(stable_now),
                    "need": int(CENTER_MIN_FRAMES),
                    "box_frac": [CENTER_BOX_FRAC_W, CENTER_BOX_FRAC_H],
                }

                if contour_hits:
                    payload["contours"] = contour_hits
                if matched_product_ids:
                    payload["matched_product_ids"] = list(matched_product_ids)
                if poi_markers:
                    payload["poi_markers"] = poi_markers

                asyncio.run_coroutine_threadsafe(broadcast_ws(payload), main_loop)
            except Exception:
                pass


#_worker = threading.Thread(target=infer_loop, daemon=True)
#_worker.start()

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

@app.get("/contours", response_class=HTMLResponse)
async def contours_page(request: Request):
    return templates.TemplateResponse("contours.html", {"request": request})

@app.get("/rejects", response_class=HTMLResponse)
async def rejects_page(request: Request):
 return templates.TemplateResponse("rejects.html", {"request": request})



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
    show_masks: Optional[bool] = None
    show_boxes: Optional[bool] = None
    contour_match_enabled: Optional[bool] = None
    contour_match_class: Optional[str] = None
    contour_match_iop: Optional[float] = None
    active_contour_ids: Optional[List[str]] = None
    active_contour_id: Optional[str] = None

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
    if body.show_masks is not None: config.show_masks = bool(body.show_masks)
    if body.show_boxes is not None: config.show_boxes = bool(body.show_boxes)
    if body.contour_match_enabled is not None: config.contour_match_enabled = bool(body.contour_match_enabled)
    if body.contour_match_class is not None:   config.contour_match_class = str(body.contour_match_class)
    if body.contour_match_iop is not None:     config.contour_match_iop = float(body.contour_match_iop)
    if body.active_contour_ids is not None:  config.active_contour_ids = [str(cid).strip() for cid in body.active_contour_ids if str(cid).strip()]
    if body.active_contour_id is not None: config.active_contour_id = body.active_contour_id or None 
    if body.active_contour_id is None: config.active_contour_id = None

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
    names = m.names if hasattr(m, "names") else {}
    has_masks = getattr(res, "masks", None) is not None and getattr(res.masks, "xy", None) is not None
    masks_xy = res.masks.xy if has_masks else None

    if res.boxes is not None and len(res.boxes):
        for k in range(len(res.boxes)):
            cls = int(res.boxes.cls[k].item())
            label = names[cls] if isinstance(names, dict) else str(cls)
            confv = float(res.boxes.conf[k].item() if res.boxes.conf is not None else 0.0)
            x1, y1, x2, y2 = map(int, res.boxes.xyxy[k].tolist())

            item = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": label, "conf": confv
            }

            # Voeg polygon-mask toe wanneer beschikbaar (YOLOv8-seg)
            if masks_xy is not None and k < len(masks_xy) and masks_xy[k] is not None:
                # masks_xy[k] is een Nx2 array [[x,y], ...] in afbeeldingspixels
                poly = masks_xy[k]
                # zorg dat het JSON-serialiseerbaar is (lijst van [x,y])
                item["mask"] = [[float(p[0]), float(p[1])] for p in poly]

            out.append(item)

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

                # korte, consistente naam: <class>_<yolov8X[-seg]>.pt
                base_key = _short_base_key(base_weights)  # bv. 'yolov8n' of 'yolov8n-seg'
                short_name = f"{class_name}_{base_key}.pt"

                # Optioneel: stuur seg-modellen standaard naar runs/segment i.p.v. runs/exported
                # (alleen als export_dir leeg gelaten wordt aan de UI-kant)
                # if export_dir is None:
                #     out_dir = Path("runs/segment" if "-seg" in base_key else "runs/exported")
                #     out_dir.mkdir(parents=True, exist_ok=True)

                out_path = _safe_out_path(out_dir, short_name)
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

def _largest_filled_polygon(img_bgr: np.ndarray, min_area_ratio: float = 0.005) -> list[list[int]]:
    """
    Zoek grootste gevulde component i.p.v. randen.
    Werkt op lastige achtergronden met Otsu + morfologie.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold (automatisch)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Kies 'foreground' richting (soms wit=achtergrond):
    if cv2.countNonZero(th) > 0.5 * W * H:
        th = cv2.bitwise_not(th)

    # Morfologisch sluiten → gaatjes dicht; dan openen → ruis weg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_area_ratio * (W * H):
        # te klein → waarschijnlijk ruis
        return []

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)  # 1% van de omtrek
    return [[int(p[0][0]), int(p[0][1])] for p in approx]


def _largest_polygon_from_mask(mask_bin: np.ndarray, epsilon_frac: float = 0.01) -> list[list[int]]:
    # Neem grootste contour van binaire mask en approximeer
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_frac * peri, True)
    return [[int(p[0][0]), int(p[0][1])] for p in approx]

def _auto_contour_from_frame(img: np.ndarray, target_class: Optional[str]) -> tuple[list[list[float]], int, int]:
    """
    Retourneer (polygon01, W, H). Probeert eerst seg-masks; anders Canny->largest contour.
    target_class: None => alle classes toegestaan; anders filter op label.
    """
    H, W = img.shape[:2]
    m = get_primary_model()
    if m is None:
        # Fallback: Canny
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ed = cv2.Canny(gray, 60, 120)
        poly_px = _largest_polygon_from_mask(ed)
        poly01 = [[max(0,min(1,x/W)), max(0,min(1,y/H))] for x,y in poly_px]
        return (poly01, W, H)

    try:
        res = m.predict(img, conf=max(0.2, config.conf), iou=config.iou, imgsz=config.imgsz, verbose=False)[0]
    except Exception:
        res = None

    # 1) als masks beschikbaar: bouw binaire mask per matchende class, kies grootste
    if res is not None and getattr(res, "masks", None) is not None and res.masks is not None:
        names = getattr(m, "names", {})
        best_area = 0
        best_poly01: list[list[float]] = []
        for i, poly in enumerate(res.masks.xy or []):
            if poly is None or len(poly) < 3:
                continue
            lab = names.get(int(res.boxes.cls[i].item()), str(int(res.boxes.cls[i].item()))) if isinstance(names, dict) else str(int(res.boxes.cls[i].item()))
            if target_class and str(lab).lower() != str(target_class).lower():
                continue
            # mask vullen om grootste-contour te kunnen approximeren
            canvas = np.zeros((H, W), np.uint8)
            pts = np.asarray(poly, np.int32).reshape(-1,1,2)
            cv2.fillPoly(canvas, [pts], 255)
            poly_px = _largest_polygon_from_mask(canvas)
            area = cv2.contourArea(np.asarray(poly_px, np.int32)) if len(poly_px) >= 3 else 0
            if area > best_area and len(poly_px) >= 3:
                best_area = area
                best_poly01 = [[max(0,min(1,x/W)), max(0,min(1,y/H))] for x,y in poly_px]
        if best_poly01:
            return (best_poly01, W, H)

    # 2) fallback: Canny grootste contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ed = cv2.Canny(gray, 60, 120)
    poly_px = _largest_filled_polygon(img)
    poly01 = [[max(0, min(1, x / W)), max(0, min(1, y / H))] for x, y in poly_px]
    return (poly01, W, H)

from fastapi import UploadFile, File, Form

@app.post("/api/contours/auto")
async def api_contours_auto(
    target_class: Optional[str] = Form(None),
    file: UploadFile | None = File(None)  # ← optioneel
):
    # 1) Als er een file is, gebruik die
    if file is not None:
        try:
            content = await file.read()
        finally:
            await file.close()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)
    else:
        # 2) Anders fallback naar het live raw frame
        global latest_raw_frame
        if latest_raw_frame is None:
            return JSONResponse({"error": "No frame"}, status_code=409)
        img = latest_raw_frame

    try:
        poly01, W, H = _auto_contour_from_frame(img, target_class)
        if len(poly01) < 3:
            return JSONResponse({"error": "No polygon found"}, status_code=404)
        return {"ok": True, "width": W, "height": H, "polygon01": poly01}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

router = APIRouter()

# --- Type -> product mapping helper (gebruik properties.type_key of properties.type) ---
def _product_id_by_type_key(type_key: str, ref_file: str | None = None) -> Optional[str]:
    """
    Zoek het product-ID op basis van 'name' in products.json.
    - Case-insensitive
    - Negeert underscores
    - Probeert eerst type_key, dan ref_file-stem (zonder extensie)
    - Laat lichte variaties toe (zoals '9T9', 'T9_dark', 'T13v2')
    """
    def _normalize(s: str) -> str:
        return str(s).strip().lower().replace("_", "").replace("-", "")

    key = _normalize(type_key)
    alt = _normalize(Path(ref_file).stem if ref_file else "")
    if not key and not alt:
        return None

    best_match = None
    for p in _load_products():
        pname = _normalize(p.get("name", ""))
        if not pname:
            continue

        # --- Exact match ---
        if pname == key or pname == alt:
            return p.get("id")

        # --- Deelstring match (T9 <-> T9dark, 9T9 <-> T9, T13v2 <-> T13) ---
        if pname in key or key in pname or pname in alt or alt in pname:
            best_match = p.get("id")

        # --- Fuzzy letter/cijfer fallback (negeert 't') ---
        elif pname.replace("t", "") == key.replace("t", "") or pname.replace("t", "") == alt.replace("t", ""):
            best_match = p.get("id")

    return best_match


# ===[ Multi-ref helpers voor contour-match ]=================================
REF_DIR = Path("./ref_contours")
REF_OK_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _ref_type_prefix_from_stem(stem: str) -> str:
    """
    Neem een type-prefix uit een bestandsnaam-stem: 'T9_02' -> 'T9', 'T13dark' -> 'T13'.
    Valt terug op de volledige stem als er geen duidelijke scheiding is.
    """
    m = re.match(r"([A-Za-z0-9]+)", stem)
    return m.group(1) if m else stem

def _iter_ref_files(prefix: str | None = None) -> list[Path]:
    """
    Vind alle referentiefoto's in ./ref_contours.
    - Als prefix opgegeven is, filter op die prefix (case-insensitive).
    - Anders: alle bestanden.
    """
    if not REF_DIR.exists():
        return []
    files = [p for p in REF_DIR.iterdir()
             if p.is_file() and p.suffix.lower() in REF_OK_SUFFIXES]
    if prefix:
        px = prefix.lower()
        files = [p for p in files if _ref_type_prefix_from_stem(p.stem).lower() == px]
    # deterministische volgorde
    files.sort(key=lambda p: p.name.lower())
    return files

def _list_ref_type_keys() -> list[str]:
    """Unieke type-prefixes die in ref_contours voorkomen, gesorteerd."""
    keys = set()
    for p in _iter_ref_files():
        keys.add(_ref_type_prefix_from_stem(p.stem))
    return sorted(keys)

def _load_ref_contour_from_file(path: Path) -> np.ndarray | None:
    ref = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if ref is None:
        return None
    try:
        cnt = _find_outer_contour_detailed(ref)
    except Exception:
        cnt = None
    return cnt

def _find_hole_contour(gray: np.ndarray, roi=None):
    return _find_outer_contour_detailed(gray)

# ---------- Helpers ----------
def _read_image(file_bytes: bytes) -> np.ndarray:
    img = cv.imdecode(np.frombuffer(file_bytes, np.uint8), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Kon afbeelding niet decoderen")
    return img

def simplify(c, frac=0.005):
    peri = cv.arcLength(c, True)
    return cv.approxPolyDP(c, max(1.0, frac*peri), True)

def _draw_cnt(img_gray, cnt, name):
    vis = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    cv.drawContours(vis, [cnt], -1, (0,255,0), 2)
    cv.imwrite(f"debug/{name}.png", vis)

def _mask_from_contours(contours: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    m = np.zeros(size, np.uint8)
    if contours:
        cv.drawContours(m, contours, -1, 255, thickness=-1)
    return m

def _draw_debug_contour(img_gray: np.ndarray, cnt, name: str):
    vis = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    if isinstance(cnt, (list, tuple)):
        cv.drawContours(vis, cnt, -1, (0, 255, 0), 1, lineType=cv.LINE_AA)
    else:
        cv.drawContours(vis, [cnt], -1, (0, 255, 0), 1, lineType=cv.LINE_AA)
    os.makedirs("debug_contours", exist_ok=True)
    cv.imwrite(os.path.join("debug_contours", f"{name}.png"), vis)

def _center_from_contour(cnt: np.ndarray) -> tuple[float,float,float]:
    # center+radius van buitenrand
    (cx, cy), r = cv.minEnclosingCircle(cnt.astype(np.float32))
    return float(cx), float(cy), float(r)

def _annulus_polar(gray: np.ndarray, cx: float, cy: float,
                   r1: float, r2: float,
                   thetas: int = 720, min_rows: int = 32) -> np.ndarray:
    """
    Unwrap annulus [r1, r2] naar een (rows, cols) beeld in polaire coördinaten.
    rows ≈ r2; daarna exact de band r1..r2 uitsnijden.
    """
    H = max(1, int(np.ceil(r2)) + 1)  # resolutie over radius-as
    polar = cv.warpPolar(
        gray, (thetas, H),
        (cx, cy), r2,
        cv.WARP_POLAR_LINEAR | cv.WARP_FILL_OUTLIERS
    )
    y1 = int(max(0, np.floor(r1)))
    y2 = int(min(H, np.ceil(r2)))
    band = polar[y1:y2, :]  # (rows, thetas)

    if band.shape[0] < min_rows:
        band = cv.resize(band, (thetas, max(min_rows, 1)), interpolation=cv.INTER_LINEAR)
    return band



def _theta_signature(polar_band: np.ndarray) -> np.ndarray:
    sig = polar_band.mean(axis=0).astype(np.float32)
    sig -= sig.mean()
    n = float(np.linalg.norm(sig) + 1e-6)
    return sig / n

def _ncc_circular(a: np.ndarray, b: np.ndarray) -> float:
    """Max. circulaire correlatie in [-1,1]; a/b zijn zero-mean en unit-norm."""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    n = int(a.size)
    A = np.fft.rfft(a); B = np.fft.rfft(b)
    xcorr = np.fft.irfft(A * np.conj(B), n=n)  # géén extra /n hier
    val = float(np.max(xcorr))
    return max(-1.0, min(1.0, val))



@lru_cache(maxsize=128)
def _load_ref_features(path_str: str):
    path = Path(path_str)
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    cnts = _find_outer_contour_detailed(img)
    cnt = cnts[0] if isinstance(cnts, list) else cnts
    cx, cy, r = _center_from_contour(cnt)

    A = _theta_signature(_annulus_polar(img, cx, cy, 0.75*r, 0.95*r, thetas=1440, min_rows=64))
    B = _theta_signature(_annulus_polar(img, cx, cy, 0.50*r, 0.70*r, thetas=1440, min_rows=64))
    C = _theta_signature(_annulus_polar(img, cx, cy, 0.30*r, 0.45*r, thetas=1440, min_rows=64))

    return {"cnt": cnt, "center": (cx, cy, r), "sigA": A, "sigB": B, "sigC": C}

def _live_features(gray: np.ndarray, cnt_live: np.ndarray):
    cx, cy, r = _center_from_contour(cnt_live)
    A = _theta_signature(_annulus_polar(gray, cx, cy, 0.75*r, 0.95*r, thetas=1440, min_rows=64))
    B = _theta_signature(_annulus_polar(gray, cx, cy, 0.50*r, 0.70*r, thetas=1440, min_rows=64))
    C = _theta_signature(_annulus_polar(gray, cx, cy, 0.30*r, 0.45*r, thetas=1440, min_rows=64))
    return (cx, cy, r), (A, B, C)


def _combined_score(cnt_ref, cnt_live, ref_feats, live_sigs,
                    w_shape=0.2, w_polar=0.8):
    s_shape = cv.matchShapes(cnt_ref, cnt_live, cv.CONTOURS_MATCH_I1, 0.0)

    A_ref, B_ref, C_ref = ref_feats["sigA"], ref_feats["sigB"], ref_feats["sigC"]
    A, B, C = live_sigs
    nccA = _ncc_circular(A, A_ref)
    nccB = _ncc_circular(B, B_ref)
    nccC = _ncc_circular(C, C_ref)
    ncc  = (nccA + nccB + nccC) / 3.0

    s_polar = 1.0 - ncc
    score = w_shape * float(s_shape) + w_polar * float(s_polar)

    # print(f"NCC: A={nccA:.3f} B={nccB:.3f} C={nccC:.3f}  mean={ncc:.3f}  shape={s_shape:.4f}  score={score:.3f}")
    return float(score), float(ncc), float(s_shape)




@lru_cache(maxsize=16)
def _load_ref_contour(name: str) -> np.ndarray:
    path = f"./ref_contours/{name}.png"
    ref_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if ref_gray is None:
        raise FileNotFoundError(f"Referentie '{name}' niet gevonden op {path}")
    # Gebruik dezelfde finder zodat ref==live consistent is:
    return _find_outer_contour_detailed(ref_gray)



def _list_ref_names() -> list[str]:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for p in glob(str(REF_DIR / "*.png")) + glob(str(REF_DIR / "*.jpg")) + glob(str(REF_DIR / "*.jpeg")):
        names.append(Path(p).stem)
    # unieke, stabiele volgorde
    return sorted(list(dict.fromkeys(names)))


def _find_outer_contour_detailed(
    gray: np.ndarray,
    *,
    focus_center: bool = True,
    min_area_frac: float = 0.001,
    max_area_frac: float = 0.80,
    simplify_eps_frac: float = 0.0000000000001,
    debug_name: str | None = None
) -> np.ndarray:
    """
    Super-gedetailleerde naafcontour:
    - gebruikt Canny + Scharr
    - minimale smoothing
    - geen vereenvoudiging
    - behoudt elk tandje en boutrandje
    """
    if gray.ndim != 2:
        raise ValueError("Grayscale beeld verwacht (1 kanaal).")

    H, W = gray.shape[:2]
    img_area = float(W * H)

    # --- Minder smoothing, meer detail ---
    g = cv.GaussianBlur(gray, (3,3), 0)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    # --- Scharr + Canny combineren ---
    gx = cv.Scharr(g, cv.CV_32F, 1, 0)
    gy = cv.Scharr(g, cv.CV_32F, 0, 1)
    mag = cv.magnitude(gx, gy)
    mag = cv.convertScaleAbs(mag)
    edges_scharr = cv.Canny(mag, 50, 200, apertureSize=3, L2gradient=True)

    edges_adapt = cv.adaptiveThreshold(
        g, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 11, -2
    )

    edges = cv.bitwise_or(edges_scharr, edges_adapt)

    # --- Minimalistische morfologie (sluit minuscule gaten) ---
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, k, iterations=1)

    # --- Contouren zoeken ---
    cnts, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("Geen contouren gevonden")

    # --- Centrumfilter (naaf in midden) ---
    cx, cy = W * 0.5, H * 0.5
    kept = []
    for c in cnts:
        a = cv.contourArea(c)
        if a < (min_area_frac * img_area) or a > (max_area_frac * img_area):
            continue
        M = cv.moments(c)
        if M["m00"] == 0:
            continue
        x = M["m10"]/M["m00"]; y = M["m01"]/M["m00"]
        if focus_center and np.hypot((x-cx)/W, (y-cy)/H) > 0.5:
            continue
        kept.append(c)

    if not kept:
        kept = [max(cnts, key=cv.contourArea)]

    # --- Combineer alles naar één contour via mask ---
    mask = np.zeros_like(gray, np.uint8)
    cv.drawContours(mask, kept, -1, 255, 1)
    ext, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not ext:
        raise ValueError("Kon geen buitencontour uit masker halen")
    combined = max(ext, key=cv.contourArea)

    # --- Debug ---
    if debug_name:
        dbg = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        cv.drawContours(dbg, [combined], -1, (0, 255, 0), 1, lineType=cv.LINE_AA)
        os.makedirs("debug_contours", exist_ok=True)
        cv.imwrite(f"debug_contours/{debug_name}_maxdetail.png", dbg)

    return combined




def _as_contour_pts(arr: np.ndarray) -> np.ndarray:
    """
    Normaliseer naar shape (N,1,2) float32/int32 voor cv.moments / cv.contourArea.
    """
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[1] == 1 and a.shape[2] == 2:
        pass
    elif a.ndim == 2 and a.shape[1] == 2:
        a = a.reshape(-1, 1, 2)
    else:
        raise ValueError(f"Ongeldige contourvorm: shape={a.shape}, dtype={a.dtype}")
    # moments werkt met float32 of int32
    if a.dtype not in (np.float32, np.int32):
        a = a.astype(np.float32, copy=False)
    return a

def _centroid(cnt: np.ndarray) -> Tuple[float, float]:
    try:
        pts = _as_contour_pts(cnt)
        M = cv.moments(pts)
    except Exception:
        # fallback: probeer als binaire image
        img = np.asarray(cnt)
        if img.ndim == 2:
            M = cv.moments(img.astype(np.uint8))
        else:
            raise

    if abs(M.get("m00", 0.0)) < 1e-6:
        pts2 = pts.reshape(-1, 2)
        x = float(np.mean(pts2[:, 0] if len(pts2) else [0]))
        y = float(np.mean(pts2[:, 1] if len(pts2) else [0]))
        return x, y
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])

def _mask_from_contour(cnt: np.ndarray, size: Tuple[int,int]) -> np.ndarray:
    m = np.zeros(size, np.uint8)
    cv.drawContours(m, [cnt], -1, 255, thickness=-1)
    return m

def _align_for_overlap(
    ref_contours: list[np.ndarray] | np.ndarray,
    live_contours: list[np.ndarray] | np.ndarray,
    out_size: int = 768
) -> tuple[np.ndarray, np.ndarray]:
    """Rudimentaire schaal/shift-align op basis van area & centroid, daarna masks vullen."""
    def _as_list(c):
        return c if isinstance(c, list) else [c]

    ref_list = _as_list(ref_contours)
    live_list = _as_list(live_contours)

    # bbox/area/centroid op gevulde mask-resolutie
    # neem canvas groot genoeg om detail te behouden
    W = H = out_size

    # maak ruwe maskers om area/centroid te schatten (op 'native' schaal)
    # eerst maximale extents inschatten
    def _extents(contours):
        xs, ys = [], []
        for c in contours:
            pts = c.reshape(-1, 2)
            xs.append(pts[:, 0]); ys.append(pts[:, 1])
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    rx1, ry1, rx2, ry2 = _extents(ref_list)
    lx1, ly1, lx2, ly2 = _extents(live_list)

    # schaal factor via (gevulde) area verhouding
    ref_area = sum(max(cv.contourArea(c), 1.0) for c in ref_list)
    live_area = sum(max(cv.contourArea(c), 1.0) for c in live_list)
    scale = float(np.sqrt(ref_area / max(live_area, 1e-6)))

    # schaal live-punten
    live_scaled = [ (c.astype(np.float32) * scale) for c in live_list ]

    # centroiden
    def _centroid_list(contours):
        m = np.zeros((H, W), np.uint8)
        # voorlopig zonder extra verschuiving tekenen (we verschuiven pas na centroid)
        # normaliseer naar canvas-doorrekenen: we passen pas shift toe later
        # om een robuuste centroid te krijgen op punten, gebruik moments van punten:
        xs, ys = [], []
        for c in contours:
            p = c.reshape(-1,2)
            xs.append(p[:,0]); ys.append(p[:,1])
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        return float(xs.mean()), float(ys.mean())

    cxr, cyr = _centroid_list(ref_list)
    cxl, cyl = _centroid_list(live_scaled)

    shift = np.array([[[cxr - cxl, cyr - cyl]]], dtype=np.float32)
    live_aligned = [ (c + shift).astype(np.int32) for c in live_scaled ]

    # center beide op canvas
    # (optioneel: je kan ook normaliseren op basis van ref-centroid → canvasmidden)
    rcx, rcy = cxr, cyr
    canv_center = np.array([[[W/2 - rcx, H/2 - rcy]]], np.float32)
    ref_centered  = [ (c.astype(np.float32) + canv_center).astype(np.int32) for c in ref_list ]
    live_centered = [ (c.astype(np.float32) + canv_center).astype(np.int32) for c in live_aligned ]

    ref_mask  = _mask_from_contours(ref_centered,  (H, W))
    live_mask = _mask_from_contours(live_centered, (H, W))
    return ref_mask, live_mask

def _compute_overlap(ref_mask: np.ndarray, live_mask: np.ndarray) -> float:
    inter = np.logical_and(ref_mask>0, live_mask>0).sum()
    uni   = np.logical_or (ref_mask>0, live_mask>0).sum()
    if uni == 0:
        return 0.0
    return float(inter/uni)  # IoU

def fit_radius(cnt: np.ndarray) -> float:
    (x, y), r = cv.minEnclosingCircle(cnt.astype(np.float32))
    return float(r)

# ---------- Endpoint ----------
CAPTURE_DIR = Path("./captures")
CAPTURE_DIR.mkdir(exist_ok=True)

_last_frame_bgr = None  # wordt elders (bijv. in infer_loop) steeds geüpdatet

def update_last_frame(frame):
    global _last_frame_bgr
    _last_frame_bgr = frame.copy()

@router.post("/api/capture")
async def api_capture():
    """Slaat het laatste frame op zonder overlays/maskers/contours."""
    global _last_frame_bgr
    if _last_frame_bgr is None:
        raise HTTPException(400, "Geen frame beschikbaar om op te slaan")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = CAPTURE_DIR / f"capture_{ts}.jpg"
    cv2.imwrite(str(out_path), _last_frame_bgr)
    return {"ok": True, "path": str(out_path)}


@router.post("/api/contour-match")
async def contour_match(
    file: UploadFile = File(...),
    ref_name: str = Form("", description="Optioneel: type/prefix (bv. T8, T13). Leeg = alle types"),
    score_thresh: float = Form(0.02, description="Max. matchShapes score (lager is beter)"),
    iou_thresh: float = Form(0.80, description="Min. IoU na align"),
    roi: str = Form("", description="Optioneel ROI 'x,y,w,h' om het gat te zoeken")
):
    """
    Vergelijk live 'gat'-contour met ÓF:
      - alle ref-afbeeldingen die bij 'ref_name' (prefix) horen,
      - ÓF (als ref_name leeg is) de best scorende variant in élk type,
        en kies vervolgens de globale winnaar.
    """
    # 1) Live input lezen
    try:
        bytes_ = await file.read()
        gray = _read_image(bytes_)
    except Exception as e:
        raise HTTPException(400, f"Afbeelding ongeldig: {e}")

    # 2) ROI (optioneel)
    parsed_roi = None
    if roi:
        try:
            x, y, w, h = map(int, roi.split(","))
            parsed_roi = (x, y, w, h)
        except Exception:
            raise HTTPException(400, "ROI formaat moet 'x,y,w,h' zijn")

    # 3) Live contour vinden
    try:
        cnt_live = _find_outer_contour_detailed(gray, debug_name="live_input")
        _draw_debug_contour(gray, cnt_live, "live_input")
        _, live_sigs = _live_features(gray, cnt_live)
    except Exception as e:
        raise HTTPException(422, f"Kon live gat-contour niet bepalen: {e}")

    # 4) Kandidaten bepalen (één type of alle types)
    if ref_name:
        type_keys = [ref_name]  # alleen dit type/prefix
    else:
        type_keys = _list_ref_type_keys()
        if not type_keys:
            raise HTTPException(404, "Geen referentiebestanden gevonden in ./ref_contours")

    per_type_best = []   # beste per type: {type_key, ref_file, score, iou, center_dist, match}
    global_best = None   # overall beste (laagste score)

    for tkey in type_keys:
        candidates = _iter_ref_files(tkey)
        if not candidates:
            # geen files voor dit type → sla type over
            continue

        best_for_type = None
        for ref_path in candidates:
            try:
                ref_feats = _load_ref_features(str(ref_path))
            except Exception:
                continue

            cnt_ref = ref_feats["cnt"]

            # gecombineerde score (lager = beter), ncc (hoger = beter), shape-info
            score, ncc, s_shape = _combined_score(cnt_ref, cnt_live, ref_feats, live_sigs)

            # extra debug: IoU na simpele align + centroidafstand
            ref_mask, live_mask = _align_for_overlap(cnt_ref, cnt_live, out_size=768)
            iou = _compute_overlap(ref_mask, live_mask)

            # centroidafstand (alleen info)
            M1 = cv.moments(cnt_ref);
            M2 = cv.moments(cnt_live)
            cx1 = M1["m10"] / max(M1["m00"], 1e-6);
            cy1 = M1["m01"] / max(M1["m00"], 1e-6)
            cx2 = M2["m10"] / max(M2["m00"], 1e-6);
            cy2 = M2["m01"] / max(M2["m00"], 1e-6)
            cdist = float(np.hypot(cx1 - cx2, cy1 - cy2))

            rec = {
                "type_key": tkey,
                "ref_file": ref_path.name,
                "score": float(score),  # onze gecombineerde kost (lager = beter)
                "ncc": float(ncc),  # 0..1 (hoger = beter)
                "shape": float(s_shape),  # ter debug
                "iou": float(iou),  # ter debug
                "center_dist_px": cdist,  # ter debug
            }
            if (best_for_type is None) or (rec["score"] < best_for_type["score"]):
                best_for_type = rec

        if best_for_type:
            # drempels toepassen voor 'match' boolean
            best_for_type["match"] = (best_for_type["score"] <= score_thresh) and (best_for_type["iou"] >= iou_thresh)
            per_type_best.append(best_for_type)
            if (global_best is None) or (best_for_type["score"] < global_best["score"]):
                global_best = best_for_type

    if not per_type_best:
        raise HTTPException(500, "Geen geldige referentiecontouren gevonden/bruikbaar")

    # thresholds (maak desnoods configurabel)
    SCORE_MAX = float(score_thresh)  # bv. 0.20 als startpunt
    NCC_MIN = 0.70  # polar correlatie
    IOU_MIN = 0.60  # IoU alleen informatief/veiligheidsnet
    MARGIN = 0.18  # #1 moet ≥20% beter zijn dan #2

    per_type_best.sort(key=lambda r: r["score"])
    global_best = per_type_best[0]
    second_best = per_type_best[1] if len(per_type_best) > 1 else None

    ok_by_thresh = (global_best["score"] <= SCORE_MAX) and (global_best["ncc"] >= NCC_MIN) and (
                global_best["iou"] >= IOU_MIN)
    ok_by_iou = (global_best.get("iou", 0.0) >= IOU_MIN)

    # relatieve marge t.o.v. #2
    ok_by_margin = True
    if second_best:
        rel = (second_best["score"] - global_best["score"]) / max(second_best["score"], 1e-6)
        ok_by_margin = (rel >= MARGIN)

    # soepele combinatieregel:
    # - primaire: score & ncc ok EN iou ok
    # - fallback A: iou heel hoog (≥0.90) EN score < 0.40 (licht ruim)
    # - fallback B: score is duidelijk #1 (margin) EN ncc ≥ 0.65
    match_flag = bool(
        (global_best["score"] <= SCORE_MAX and
         global_best.get("ncc", 0.0) >= NCC_MIN and
         ok_by_iou)
        or
        (global_best.get("iou", 0.0) >= 0.90 and global_best["score"] <= 0.40)
        or
        (ok_by_margin and global_best.get("ncc", 0.0) >= 0.65)
    )

    global_best["match"] = match_flag

    # 5) Antwoord
    # Bewaar backward compat: 'best' bevat de globale winnaar;
    # 'all' bevat alle "beste per type".
    # thresholds teruggeven helpt voor debuggen aan de UI-kant.
    return {
        "ok": True,
        "best": {
            "type_key": global_best["type_key"],
            "ref": global_best["ref_file"],
            "score": global_best["score"],  # gecombineerde kost
            "ncc": global_best["ncc"],  # polar correlatie
            "shape": global_best["shape"],  # matchShapes
            "iou": global_best["iou"],
            "center_dist_px": global_best["center_dist_px"],
            "match": global_best["match"],
        },
        "all": per_type_best,
        "thresholds": {
            "score_max": SCORE_MAX,
            "ncc_min": NCC_MIN,
            "iou_min": IOU_MIN,
            "margin_min": MARGIN
        },
    }


DEBUG_DIR = "./debug_contours"
os.makedirs(DEBUG_DIR, exist_ok=True)


@app.post("/api/contours/ref/upload")
async def upload_ref_contour(
    file: UploadFile = File(...),
    name: str | None = Form(None)   # optioneel: gewenste naam zonder extensie (bv. "T9")
):
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # valideer extensie
    orig_ext = Path(file.filename).suffix.lower()
    if orig_ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Ongeldige extensie")

    # bepaal bestandsnaam
    stem = (name or Path(file.filename).stem).strip()
    if not stem:
        raise HTTPException(status_code=400, detail="Naam ontbreekt")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)

    out_path = REF_DIR / f"{safe_stem}{orig_ext}"

    # schrijf bestand
    data = await file.read()
    out_path.write_bytes(data)

    # klaar
    return {
        "ok": True,
        "filename": out_path.name,
        "path": str(out_path.resolve()),
        "ref_name": safe_stem
    }
# ------------------------------
# Collections + Products JSON storage
# ------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PRODUCTS_DB = DATA_DIR / "products.json"
COLLECTIONS_DB = DATA_DIR / "collections.json"
CONTOURS_DB = DATA_DIR / "contours.json"
POIS_DB = DATA_DIR / "pois.json"
ANNS_DB = DATA_DIR / "annotations.json"

def _load_annotations() -> list[dict]:
    return _load_json(ANNS_DB)

def _save_annotations(items: list[dict]) -> None:
    _save_json(ANNS_DB, items)

class Box01(BaseModel):
    x: float
    y: float
    w: float
    h: float
    label: str

class Point(BaseModel):
    x: float
    y: float

class AnnUpsert(BaseModel):
    key: str
    name: Optional[str] = None
    w: int
    h: int
    boxes01: List[Box01] = Field(default_factory=list)
    masks01: List[List[Point]] = Field(default_factory=list)

@app.get("/api/ann")
async def api_ann_get(key: str):
    items = _load_annotations()
    for it in items:
        if it.get("key") == key:
            return it
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.post("/api/ann")
async def api_ann_upsert(payload: AnnUpsert):
    items = _load_annotations()
    found = False
    for i, it in enumerate(items):
        if it.get("key") == payload.key:
            items[i] = { **it, **payload.model_dump() }
            found = True
            break
    if not found:
        items.append(payload.model_dump())
    _save_annotations(items)
    return {"ok": True}


def _load_pois() -> list[dict]: return _load_json(POIS_DB)
def _save_pois(items: list[dict]): _save_json(POIS_DB, items)

class POIItem(BaseModel):
    x01: float
    y01: float
    z: Optional[float] = None
    expected_label: str
    required: int = 1
    radius01: float = 0.02  # ~2% van min(W,H)

class POISet(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    product_id: Optional[str] = None      # POI’s horen doorgaans bij product
    contour_id: Optional[str] = None      # optioneel: alleen geldig binnen deze contour
    items: List[POIItem] = Field(default_factory=list)

@app.get("/api/pois")
async def api_pois_list(product_id: Optional[str] = None, contour_id: Optional[str] = None):
    items = _load_pois()
    if product_id:
        items = [it for it in items if it.get("product_id") == product_id]
    if contour_id:
        items = [it for it in items if it.get("contour_id") == contour_id]
    return items

@app.post("/api/pois")
async def api_pois_create(p: POISet):
    items = _load_pois()
    items.append(p.model_dump())
    _save_pois(items)
    return p

@app.put("/api/pois/{pid}")
async def api_pois_update(pid: str, payload: dict):
    items = _load_pois()
    for i, it in enumerate(items):
        if it.get("id") == pid:
            allowed = {"name","product_id","contour_id","items"}
            it.update({k:v for k,v in payload.items() if k in allowed})
            items[i] = it
            _save_pois(items)
            return it
    return JSONResponse({"error":"Not found"}, status_code=404)

@app.delete("/api/pois/{pid}")
async def api_pois_delete(pid: str):
    items = _load_pois()
    new = [it for it in items if it.get("id") != pid]
    if len(new) == len(items):
        return JSONResponse({"error":"Not found"}, status_code=404)
    _save_pois(new)
    return {"ok": True, "deleted": pid}

def _load_contours() -> list[dict]:
    return _load_json(CONTOURS_DB)

def _save_contours(items: list[dict]) -> None:
    _save_json(CONTOURS_DB, items)

def _norm_poly(points: list[list[float]], w: int, h: int) -> list[list[float]]:
    # absolute px -> normalized [0..1]
    return [[max(0.0, min(1.0, x / float(w))), max(0.0, min(1.0, y / float(h)))] for x, y in points]

def _denorm_poly(points01: list[list[float]], w: int, h: int) -> list[list[int]]:
    # normalized -> absolute px (ints)
    return [[int(round(x * w)), int(round(y * h))] for x, y in points01]


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

class Polygon(BaseModel):
    cls: int
    points: List[List[float]]  # pixelcoords [[x,y],...]

class SaveMasksReq(BaseModel):
    image_path: str  # bv. "/uploads/trainer/abc123.jpg" of een relatieve path
    width: int
    height: int
    polygons: List[Polygon]

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

LABELS_DIR = Path("labels")  # of wherever jouw labels staan

def _to_label_path(image_path: str) -> Path:
    p = Path(image_path)
    # zorg dat je naar de *bron*afbeelding wijst of reconstrueer het pad:
    stem = p.stem
    return LABELS_DIR / f"{stem}.txt"

@app.post("/api/trainer/save-masks")
async def save_masks(payload: SaveMasksReq):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    img_w, img_h = float(payload.width), float(payload.height)

    lines = []
    for poly in payload.polygons:
        coords = []
        for x, y in poly.points:
            xn = max(0.0, min(1.0, x / img_w))
            yn = max(0.0, min(1.0, y / img_h))
            coords.append(f"{xn:.6f}")
            coords.append(f"{yn:.6f}")
        if len(coords) >= 6:  # minimaal drie punten
            lines.append(f"{int(poly.cls)} " + " ".join(coords))

    if not lines:
        return {"ok": False, "error": "Geen geldige polygonen"}

    label_path = _to_label_path(payload.image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return {"ok": True, "label_path": str(label_path)}

class Contour(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str                      # e.g. "Type 1"
    type_key: str                  # machine label you’ll map to product type (e.g. "type1")
    image_name: Optional[str] = None  # original reference filename (optional)
    image_url: Optional[str] = None
    width: int
    height: int
    polygon01: List[List[float]]   # normalized [[x,y]...]
    product_id: Optional[str] = None

def _poly_iop(template_poly_xy: list[list[int]], mask_poly_xy: list[list[float]], w: int, h: int) -> float:
    """
    IoP = area(intersection) / area(template_polygon)
    Args are in frame coordinates (pixels).
    """
    if len(template_poly_xy) < 3 or len(mask_poly_xy) < 3:
        return 0.0
    tpl = np.zeros((h, w), dtype=np.uint8)
    msk = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(tpl, [np.asarray(template_poly_xy, dtype=np.int32)], 255)
    cv2.fillPoly(msk, [np.asarray(mask_poly_xy, dtype=np.int32)], 255)
    inter = cv2.bitwise_and(tpl, msk)
    area_tpl = int(cv2.countNonZero(tpl))
    if area_tpl <= 0:
        return 0.0
    area_inter = int(cv2.countNonZero(inter))
    return float(area_inter) / float(area_tpl + 1e-6)


@app.get("/api/contours")
async def api_list_contours():
    return _load_contours()

@app.post("/api/contours")
async def create_contour(c: Contour):
    items = _load_contours()
    items.append(c.model_dump())
    _save_contours(items)
    return c

@app.put("/api/contours/{cid}")
async def update_contour(cid: str, payload: dict):
    items = _load_contours()
    for i, it in enumerate(items):
        if it.get("id") == cid:
            it.update({k: v for k, v in payload.items()
                       if k in ("name","type_key","polygon01","width","height",
                                "image_name","image_url","product_id")})
            items[i] = it
            _save_contours(items)
            return it
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.delete("/api/contours/{cid}")
async def api_delete_contour(cid: str):
    items = _load_contours()
    new = [it for it in items if it.get("id") != cid]
    if len(new) == len(items):
        return JSONResponse({"error":"Not found"}, status_code=404)
    _save_contours(new)
    return {"ok": True, "deleted": cid}

@app.get("/contours-ui", response_class=HTMLResponse)
async def contours_ui(request: Request):
    return templates.TemplateResponse("contours.html", {"request": request})

# --- Rejects opslag ---
REJECTS_DIR = UPLOAD_ROOT / "rejects"
REJECTS_DIR.mkdir(parents=True, exist_ok=True)
REJECTS_DB = DATA_DIR / "rejects.json"

def _load_rejects() -> list[dict]:
    return _load_json(REJECTS_DB)

def _save_rejects(items: list[dict]) -> None:
    _save_json(REJECTS_DB, items)

from datetime import datetime

class RejectIn(BaseModel):
    product_id: str
    product_name: str
    mismatches: list[dict] = Field(default_factory=list)  # [{name, expected, actual}]
    expected_properties: dict[str, AllowedVal] = Field(default_factory=dict)
    present_counts: dict[str, int] = Field(default_factory=dict)

@app.post("/api/rejects")
async def api_rejects_add(payload: RejectIn):
    # 1) pak laatste frame
    global latest_frame
    frame = latest_frame
    if frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=409)

    # 2) schrijf jpg
    ts = time.time()
    stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
    fname = f"reject_{stamp}_{uuid4().hex[:8]}.jpg"
    fpath = REJECTS_DIR / fname
    try:
        ok = cv2.imwrite(str(fpath), frame)
        if not ok:
            return JSONResponse({"error": "Failed to write image"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": f"Save error: {e}"}, status_code=500)

    # 3) record in DB
    items = _load_rejects()
    rec = {
        "id": uuid4().hex,
        "ts": int(ts),
        "iso": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
        "product_id": payload.product_id,
        "product_name": payload.product_name,
        "image_url": f"/uploads/rejects/{fname}",
        "mismatches": payload.mismatches,
        "expected_properties": payload.expected_properties,
        "present_counts": payload.present_counts,
    }
    items.append(rec)
    _save_rejects(items)
    return {"ok": True, "record": rec}

@app.get("/api/rejects")
async def api_rejects_list():
    items = _load_rejects()
    # nieuwste eerst
    items.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return items

@app.get("/rejects-ui", response_class=HTMLResponse)
async def rejects_ui(request: Request):
    return templates.TemplateResponse("rejects.html", {"request": request})

class RejectsDeleteIn(BaseModel):
    ids: list[str] = Field(default_factory=list)

@app.post("/api/rejects/delete")
async def api_rejects_delete(payload: RejectsDeleteIn):
    items = _load_rejects()
    keep = [it for it in items if it.get("id") not in set(payload.ids)]
    deleted = [it.get("id") for it in items if it.get("id") not in {x.get("id") for x in keep}]
    _save_rejects(keep)
    # (optioneel: verwijder ook de bijhorende images)
    for it in items:
      if it.get("id") in deleted:
        try:
          p = Path("." + it.get("image_url")).resolve()
          if p.exists():
            p.unlink(missing_ok=True)
        except Exception:
          pass
    return {"ok": True, "deleted": deleted}

app.include_router(router)
# ------------------------------
# Lifecycle hook
# ------------------------------
@app.on_event("startup")
async def _on_startup():
    global main_loop, _infer_thread, _cv2_thread
    main_loop = asyncio.get_running_loop()

    if _infer_thread is None or not _infer_thread.is_alive():
        _infer_thread = threading.Thread(target=infer_loop, daemon=True)
        _infer_thread.start()

    if _cv2_thread is None or not _cv2_thread.is_alive():
         _cv2_thread = threading.Thread(target=cv2_loop, daemon=True)
         _cv2_thread.start()


# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    ssl_key = "key.pem" if os.path.exists("key.pem") else None
    ssl_crt = "cert.pem" if os.path.exists("cert.pem") else None

    host = "0.0.0.0"
    port = 8000
    scheme = "https" if (ssl_key and ssl_crt) else "http"
    # Toon een URL die voor andere devices op je LAN werkt:
    origin_url = f"{scheme}://{_get_lan_ip()}:{port}/"

    # Optioneel: zet NO_AUTO_OPEN=1 in je env om dit uit te zetten
    if os.getenv("NO_AUTO_OPEN") != "1":
        print(f"[INFO] Opening browser op: {origin_url}")
        _open_browser_soon(origin_url, delay=5)

    uvicorn.run(
        app, host=host, port=port,
        ssl_keyfile=ssl_key, ssl_certfile=ssl_crt
    )

