# ====================== automaticAttedance.py ======================
# Face Detection  : InsightFace buffalo_s  (D:\Attendance_Accurate\models\)
# Face Recognition: Per-student LBPH       (Trainer_<ID>.yml per student)
# Phone Detection : best.pt ONLY           (with face-region exclusion zones)
# ===================================================================
import tkinter as tk
from tkinter import *
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time
import pyttsx3
import threading
from collections import deque
from ultralytics import YOLO

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("WARNING: insightface not installed. Run: pip install insightface onnxruntime")

# ====================== BASE DIRECTORY ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================== COLOR SCHEME ======================
PURPLE_BG    = "#4a2c82"
PURPLE_DARK  = "#3a1c72"
PURPLE_LIGHT = "#b589d6"
TEXT_COLOR   = "#ffffff"
ACCENT_COLOR = "#ffcc00"
ENTRY_BG     = "#5d3a9b"

# ====================== ALL PATHS IN ONE PLACE ======================
# --- InsightFace ---
# InsightFace appends \models\buffalo_s to INSIGHTFACE_ROOT automatically.
# So set INSIGHTFACE_ROOT to the PARENT of the "models" folder.
# Current:  D:\Attendance_Accurate\models\buffalo_s   (correct)
# To move:  change ONLY this one variable.
#   Example: INSIGHTFACE_ROOT = r"E:\MyModels"
#            → InsightFace will look in E:\MyModels\models\buffalo_s
INSIGHTFACE_ROOT = BASE_DIR  # = D:\Attendance_Accurate

TRAINIMAGELABEL_DIR = os.path.join(BASE_DIR, "TrainingImageLabel")
COMBINED_MODEL_PATH = os.path.join(TRAINIMAGELABEL_DIR, "Trainner.yml")
STUDENTDETAIL_PATH  = os.path.join(BASE_DIR, "StudentDetails", "studentdetails.csv")
ATTENDANCE_PATH     = os.path.join(BASE_DIR, "Attendance")
SAVE_FOLDER         = os.path.join(BASE_DIR, "phone_detections")

# ====================== PER-USER DATA ISOLATION ======================
# The Flask web app calls set_user_root(<UserData/username>) before invoking
# any function in this module so that each account has its OWN students,
# trained models, attendance CSVs and phone-detection screenshots. The
# functions below read these module-level globals by name, so updating
# them here is enough — no other code changes are needed.
def set_user_root(user_dir):
    global TRAINIMAGELABEL_DIR, COMBINED_MODEL_PATH
    global STUDENTDETAIL_PATH, ATTENDANCE_PATH, SAVE_FOLDER
    TRAINIMAGELABEL_DIR = os.path.join(user_dir, "TrainingImageLabel")
    COMBINED_MODEL_PATH = os.path.join(TRAINIMAGELABEL_DIR, "Trainner.yml")
    STUDENTDETAIL_PATH  = os.path.join(user_dir, "StudentDetails", "studentdetails.csv")
    ATTENDANCE_PATH     = os.path.join(user_dir, "Attendance")
    SAVE_FOLDER         = os.path.join(user_dir, "phone_detections")
    os.makedirs(TRAINIMAGELABEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STUDENTDETAIL_PATH), exist_ok=True)
    os.makedirs(ATTENDANCE_PATH, exist_ok=True)
    os.makedirs(SAVE_FOLDER, exist_ok=True)
PHONE_MODEL_PATH    = os.path.join(BASE_DIR, "best.pt")
PHONE_MODEL2_PATH   = os.path.join(BASE_DIR, "yolo11l.pt")  # second model for ensemble

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ====================== PHONE DETECTION SETTINGS ======================
# Tuned for *higher recall* — phone was being missed too often.
# If you start getting false positives again, raise PHONE_CONFIDENCE first.
PHONE_CONFIDENCE      = 0.65   # Higher threshold = fewer false positives (books, fans, etc.)
PHONE_IMG_SIZE        = 640
PHONE_CLASS           = 0      # class 0 in best.pt = mobile phone

# Size filter
PHONE_MIN_WIDTH       = 35
PHONE_MIN_HEIGHT      = 55

# Shape filter: real phones are rectangular.
# PHONE_MIN_HW_RATIO raised to 0.40 — a book/fan can be wide and flat,
# a phone must be at least 40% as tall as it is wide.
PHONE_MAX_WH_RATIO    = 2.5
PHONE_MIN_HW_RATIO    = 0.40

# Area filter: a phone box cannot cover more than 28% of the frame
# (a fan, large book, or whiteboard would cover much more).
PHONE_MAX_AREA_FRAC   = 0.28

# Face exclusion zone
FACE_OVERLAP_REJECT   = 0.65

# Temporal smoothing — require 5 of the last 10 frames (was 3) to reduce
# one-off false triggers from objects that briefly look phone-shaped.
SMOOTHING_FRAMES      = 10
MIN_CONFIRM_FRAMES    = 5
ALERT_COOLDOWN        = 4.0    # seconds between alerts

detection_history = deque(maxlen=SMOOTHING_FRAMES)
last_event_time   = 0
phone_model       = None

# ====================== TEXT TO SPEECH ======================
# IMPORTANT: pyttsx3's runAndWait() BLOCKS the calling thread until the
# sentence is finished speaking (~3-5 seconds). When called from the camera
# loop the webcam appeared to "freeze" right after the first phone alert
# because no new frame was being read for several seconds. We now run TTS
# on a daemon thread so detection keeps streaming.
_tts_lock = threading.Lock()

def text_to_speech(user_text):
    def _speak(msg):
        with _tts_lock:
            try:
                eng = pyttsx3.init()
                eng.say(msg)
                eng.runAndWait()
                try: eng.stop()
                except: pass
            except Exception as e:
                print(f"TTS Error: {e}")
    threading.Thread(target=_speak, args=(user_text,), daemon=True).start()

# ====================== LOAD PHONE MODEL ======================
def load_phone_model():
    global phone_model
    # Primary: best.pt (custom trained on phones)
    if os.path.exists(PHONE_MODEL_PATH):
        try:
            phone_model = YOLO(PHONE_MODEL_PATH)
            print(f"Phone model loaded: {PHONE_MODEL_PATH}")
        except Exception as e:
            print(f"Phone model load failed: {e}"); phone_model = None
    else:
        print(f"WARNING: Phone model not found at {PHONE_MODEL_PATH}")

phone_model2 = None
def load_phone_model2():
    global phone_model2
    # Second model: yolo11l.pt for ensemble detection
    if os.path.exists(PHONE_MODEL2_PATH):
        try:
            phone_model2 = YOLO(PHONE_MODEL2_PATH)
            print(f"Phone model 2 loaded: {PHONE_MODEL2_PATH}")
        except Exception as e:
            print(f"Phone model 2 load failed: {e}"); phone_model2 = None
    else:
        print(f"yolo11l.pt not found — single model mode")

load_phone_model()
load_phone_model2()

# ====================== INSIGHTFACE SETUP ======================
face_app = None

def load_insightface():
    global face_app
    if not INSIGHTFACE_AVAILABLE:
        print("InsightFace not available.")
        return
    try:
        face_app = FaceAnalysis(
            name="buffalo_s",
            root=INSIGHTFACE_ROOT,   # InsightFace appends \models\buffalo_s here
            providers=["CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        resolved = os.path.join(INSIGHTFACE_ROOT, "models", "buffalo_s")
        print(f"InsightFace loaded from: {resolved}")
    except Exception as e:
        print(f"InsightFace load error: {e}")
        face_app = None

load_insightface()


def detect_faces_insightface(frame_bgr):
    """
    Returns list of (x, y, w, h) for every face detected.
    InsightFace is accurate at distance and in multi-row classrooms.
    Does NOT trigger on walls, clothes, or random objects (unlike Haar).
    """
    return [box for box, _ in get_faces_with_embeddings(frame_bgr)]


def get_faces_with_embeddings(frame_bgr):
    """
    Detect faces with InsightFace and return BOTH bounding boxes AND ArcFace
    embeddings in a single GPU/CPU pass.

    Returns list of ((x, y, w, h), embedding) tuples.
    embedding is the 512-dim ArcFace vector from buffalo_s, or None if
    InsightFace is not loaded.

    Use this in the attendance loop so you get face detection + recognition
    data from one call, instead of calling detect_faces_insightface() and then
    separately running a recognizer.
    """
    if face_app is None:
        return []
    try:
        faces = face_app.get(frame_bgr)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1 = max(0, bbox[0]); y1 = max(0, bbox[1])
            x2 = min(frame_bgr.shape[1], bbox[2])
            y2 = min(frame_bgr.shape[0], bbox[3])
            w, h = x2 - x1, y2 - y1
            if w > 25 and h > 25:
                emb = getattr(face, 'embedding', None)
                results.append(((x1, y1, w, h), emb))
        return results
    except Exception as e:
        print(f"InsightFace detection error: {e}")
        return []


# ====================== PHONE/FACE OVERLAP HELPER ======================
def _iou_overlap(boxA, boxB):
    """
    Returns what fraction of boxA is covered by boxB.
    boxA, boxB = (x, y, w, h)
    Used to reject phone detections that heavily overlap with a face region.
    """
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    areaA      = max((ax2 - ax1) * (ay2 - ay1), 1)
    return inter_area / areaA


# ====================== PER-STUDENT RECOGNIZER LOADING ======================
def load_per_student_recognizers():
    """
    Load each student's individual LBPH model.
    Trainer_1.yml  → was trained ONLY on student 1's face images
    Trainer_2.yml  → was trained ONLY on student 2's face images
    They are separate models, NOT copies of each other.

    During recognition we run every face against ALL models and pick the
    one with lowest distance. If that distance is below THRESHOLD → match.
    """
    recognizers = {}
    if not os.path.exists(TRAINIMAGELABEL_DIR):
        print("TrainingImageLabel folder not found.")
        return recognizers

    for fname in os.listdir(TRAINIMAGELABEL_DIR):
        if not (fname.startswith("Trainer_") and fname.endswith(".yml")):
            continue
        try:
            eid     = int(fname.replace("Trainer_", "").replace(".yml", ""))
            rec     = cv2.face.LBPHFaceRecognizer_create()
            rec.read(os.path.join(TRAINIMAGELABEL_DIR, fname))
            recognizers[eid] = rec
            print(f"  Loaded personal model → {fname} (student ID {eid})")
        except Exception as e:
            print(f"  Failed to load {fname}: {e}")

    print(f"Total per-student models loaded: {len(recognizers)}")
    return recognizers


def recognize_face(gray_face_roi, recognizers, student_df):
    """
    Compare a face against every student's personal model.
    Returns (enrollment_id, name, distance) for the best match,
    or (None, 'Unknown', distance) if no student matches well enough.

    THRESHOLD = 70: LBPH distance below 70 = strong match.
    Lower value = stricter (fewer false positives but may miss some).
    Higher value = looser (catches more but risks false positives).
    """
    THRESHOLD = 70

    if not recognizers:
        return None, "Unknown", 999

    try:
        face_resized = cv2.resize(gray_face_roi, (200, 200))
    except Exception:
        return None, "Unknown", 999

    best_id   = None
    best_dist = float('inf')

    for eid, rec in recognizers.items():
        try:
            _, distance = rec.predict(face_resized)
            # Each model only knows one student, so we use distance as confidence.
            # Lower distance = more confident match.
            if distance < best_dist:
                best_dist = distance
                best_id   = eid
        except Exception as e:
            print(f"Prediction error for model ID {eid}: {e}")

    if best_dist < THRESHOLD and best_id is not None:
        match = student_df[student_df["Enrollment"] == best_id]
        if not match.empty:
            return best_id, str(match["Name"].values[0]), best_dist

    return None, "Unknown", best_dist


# ====================== INSIGHTFACE EMBEDDING RECOGNITION ======================
def load_per_student_embeddings():
    """
    Load per-student ArcFace embeddings saved by trainImage.py as Embedding_<ID>.npy.
    Returns dict {enrollment_id: L2-normalized_embedding_array}.

    WHY THIS FIXES THE BEARDED-PERSON BUG:
    LBPH compares local texture patterns. Two bearded faces produce very similar
    texture patterns, so LBPH can't tell them apart. InsightFace ArcFace embeddings
    are 512-dimensional identity vectors trained on millions of faces to keep
    the SAME person's embeddings close together and DIFFERENT people's embeddings
    far apart — regardless of beard, glasses, or lighting changes.
    """
    embeddings = {}
    if not os.path.exists(TRAINIMAGELABEL_DIR):
        return embeddings
    for fname in os.listdir(TRAINIMAGELABEL_DIR):
        if not (fname.startswith("Embedding_") and fname.endswith(".npy")):
            continue
        try:
            eid = int(fname.replace("Embedding_", "").replace(".npy", ""))
            emb = np.load(os.path.join(TRAINIMAGELABEL_DIR, fname))
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings[eid] = emb
            print(f"  Loaded ArcFace embedding → {fname} (student ID {eid})")
        except Exception as e:
            print(f"  Failed to load embedding {fname}: {e}")
    print(f"Total ArcFace embeddings loaded: {len(embeddings)}")
    return embeddings


def recognize_face_embedding(face_embedding, embeddings, student_df):
    """
    Compare a face embedding against all stored student embeddings via cosine
    similarity (dot product of L2-normalized vectors).

    Returns (enrollment_id, name, similarity) or (None, 'Unknown', similarity).

    Threshold 0.35:
      >= 0.35 → same person (ArcFace typically gives 0.5-0.9 for same person)
      <  0.35 → different person (different people typically score 0.0-0.2)

    This is much more accurate than LBPH for people with similar appearance
    (beards, similar face shape) because ArcFace was trained specifically to
    preserve identity across appearance changes.
    """
    SIMILARITY_THRESHOLD = 0.35

    if not embeddings or face_embedding is None:
        return None, "Unknown", 0.0

    probe = np.array(face_embedding, dtype=np.float32).flatten()
    norm  = np.linalg.norm(probe)
    if norm == 0:
        return None, "Unknown", 0.0
    probe /= norm

    best_id  = None
    best_sim = -1.0
    for eid, stored_emb in embeddings.items():
        sim = float(np.dot(probe, stored_emb))
        if sim > best_sim:
            best_sim = sim
            best_id  = eid

    if best_sim >= SIMILARITY_THRESHOLD and best_id is not None:
        match = student_df[student_df["Enrollment"] == best_id]
        if not match.empty:
            return best_id, str(match["Name"].values[0]), best_sim

    return None, "Unknown", best_sim


# ====================== PHONE DETECTION (with face exclusion zones) ======================
def _passes_phone_filters(x1, y1, x2, y2, frame_h, frame_w, face_boxes_current):
    """
    Apply size / shape / area / face-exclusion filters to a candidate phone box.
    Returns True if the box should be kept, False if it should be discarded.
    """
    bw = x2 - x1
    bh = y2 - y1

    # Size filter
    if bw < PHONE_MIN_WIDTH or bh < PHONE_MIN_HEIGHT:
        return False

    # Aspect ratio: phone must be reasonably tall (not flat like a book spine)
    wh_ratio = bw / max(bh, 1)
    hw_ratio = bh / max(bw, 1)
    if wh_ratio > PHONE_MAX_WH_RATIO:
        return False
    if hw_ratio < PHONE_MIN_HW_RATIO:
        return False

    # Area filter: phone box can't cover too much of the frame
    frame_area = max(frame_h * frame_w, 1)
    if (bw * bh) / frame_area > PHONE_MAX_AREA_FRAC:
        return False

    # Face exclusion zone
    phone_box = (x1, y1, bw, bh)
    for fb in face_boxes_current:
        fx, fy, fw, fh = fb
        expanded_face = (
            max(0, fx - int(fw * 0.1)),
            max(0, fy - int(fh * 0.1)),
            int(fw * 1.2),
            int(fh * 1.4)
        )
        if _iou_overlap(phone_box, expanded_face) >= FACE_OVERLAP_REJECT:
            return False

    return True


def _boxes_agree(bx1, by1, bx2, by2, coco_boxes):
    """
    Check whether a best.pt candidate box overlaps with at least one COCO
    (yolo11l) detection. We use a generous IoU > 0.10 OR center-point
    containment check.
    """
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2
    for (cx1, cy1, cx2, cy2) in coco_boxes:
        # Center of best.pt box inside COCO box → clear agreement
        if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
            return True
        # IoU > 0.10 → partial spatial agreement
        inter_w = max(0, min(bx2, cx2) - max(bx1, cx1))
        inter_h = max(0, min(by2, cy2) - max(by1, cy1))
        inter_a = inter_w * inter_h
        if inter_a == 0:
            continue
        area_b = max((bx2 - bx1) * (by2 - by1), 1)
        area_c = max((cx2 - cx1) * (cy2 - cy1), 1)
        iou = inter_a / (area_b + area_c - inter_a)
        if iou > 0.10:
            return True
    return False


def detect_phone(frame, face_boxes_current=None, subject_name=""):
    """
    Detect a mobile phone in the frame using a four-layer false positive filter.

    Layer 1 — Confidence 0.65 (best.pt): raised from 0.45 to eliminate most
              casual misclassifications (books, fans, etc.)
    Layer 2 — Size + shape + area filters: phones are tall-ish rectangles that
              don't dominate the frame.
    Layer 3 — Face exclusion zones: heavy face overlap → rejected.
    Layer 4 — ENSEMBLE AGREEMENT (KEY FIX for books/fans/blades):
              When yolo11l.pt is loaded, a best.pt detection is only accepted if
              yolo11l ALSO detects a phone in roughly the same region.
              yolo11l uses COCO class 67 ("cell phone") — a book or fan blade
              will NOT be classified as class 67 by this general-purpose model,
              so it will not confirm the false positive from best.pt.
              Detections found by yolo11l alone are also accepted (it's reliable).

    face_boxes_current: list of (x,y,w,h) from InsightFace for the current frame.
    """
    global last_event_time

    display_frame      = frame.copy()
    face_boxes_current = face_boxes_current or []
    fh, fw             = frame.shape[:2]

    if phone_model is None:
        detection_history.append(0)
        return display_frame, False

    confirmed_boxes = []   # list of (x1,y1,x2,y2,conf,label) that pass ALL layers

    try:
        # ---- Step 1: Collect best.pt candidates ----
        best_candidates = []
        results = phone_model(frame, imgsz=PHONE_IMG_SIZE, conf=PHONE_CONFIDENCE,
                              classes=[PHONE_CLASS], verbose=False)
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf)
                if _passes_phone_filters(x1, y1, x2, y2, fh, fw, face_boxes_current):
                    best_candidates.append((x1, y1, x2, y2, conf))

        # ---- Step 2: Collect yolo11l COCO class-67 detections ----
        # Lower threshold for COCO model — used as confirmation, not primary.
        coco_candidates = []
        coco_raw        = []   # (x1,y1,x2,y2) for overlap check
        if phone_model2 is not None:
            r2 = phone_model2(frame, imgsz=PHONE_IMG_SIZE, conf=0.40,
                              classes=[67], verbose=False)
            if r2 and r2[0].boxes:
                for box in r2[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf)
                    if _passes_phone_filters(x1, y1, x2, y2, fh, fw, face_boxes_current):
                        coco_candidates.append((x1, y1, x2, y2, conf))
                        coco_raw.append((x1, y1, x2, y2))

        # ---- Step 3: Ensemble voting ----
        if phone_model2 is not None:
            # best.pt only accepted when yolo11l agrees
            for (x1, y1, x2, y2, conf) in best_candidates:
                if _boxes_agree(x1, y1, x2, y2, coco_raw):
                    confirmed_boxes.append((x1, y1, x2, y2, conf, "PHONE"))

            # COCO detections accepted on their own (reliable model)
            confirmed_set = {(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in confirmed_boxes}
            for (x1, y1, x2, y2, conf) in coco_candidates:
                if (x1, y1, x2, y2) not in confirmed_set:
                    confirmed_boxes.append((x1, y1, x2, y2, conf, "PHONE"))
        else:
            # No COCO model — use best.pt alone (already filtered at 0.65+)
            for (x1, y1, x2, y2, conf) in best_candidates:
                confirmed_boxes.append((x1, y1, x2, y2, conf, "PHONE"))

        # ---- Step 4: Draw confirmed detections ----
        for (x1, y1, x2, y2, conf, lbl) in confirmed_boxes:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display_frame, f"{lbl} {conf:.2f}",
                        (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    except Exception as e:
        print(f"Phone detection error: {e}")

    detected_this_frame = len(confirmed_boxes) > 0

    detection_history.append(1 if detected_this_frame else 0)
    confirmed = (sum(detection_history) >= MIN_CONFIRM_FRAMES)

    current_time = time.time()
    if confirmed and (current_time - last_event_time) >= ALERT_COOLDOWN:
        last_event_time = current_time

        # Snapshot the frame BEFORE handing the alert work to a background
        # thread. winsound.Beep() blocks for 350ms and cv2.imwrite() touches
        # the disk — running both off the camera loop is what caused the
        # webcam to "freeze" right after the first phone screenshot.
        snapshot = display_frame.copy()
        ts       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder   = os.path.join(SAVE_FOLDER, subject_name) if subject_name else SAVE_FOLDER
        fname    = f"{subject_name}_{ts}.jpg" if subject_name else f"phone_{ts}.jpg"

        def _alert_work():
            try:
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(os.path.join(folder, fname), snapshot)
                print(f"PHONE DETECTED → saved: {fname}")
            except Exception as e:
                print(f"Screenshot save failed: {e}")
            try:
                import winsound
                winsound.Beep(1900, 350)
            except Exception:
                pass
        threading.Thread(target=_alert_work, daemon=True).start()

        # text_to_speech is itself non-blocking now (runs on its own thread).
        text_to_speech("Mobile phone detected. Please put it away.")

    return display_frame, confirmed


# ====================== TIME SLOT HELPERS ======================
def get_three_time_slots(parent_window):
    ts_win = Toplevel(parent_window)
    ts_win.title("Set Attendance Time Slots")
    ts_win.geometry("500x530")
    ts_win.configure(background=PURPLE_BG)
    ts_win.resizable(False, False)

    Label(ts_win, text="SET ATTENDANCE TIME SLOTS",
          bg=PURPLE_BG, fg=ACCENT_COLOR,
          font=("Verdana", 16, "bold")).pack(pady=18)
    Label(ts_win, text="Select 3 time slots (24-hour format)",
          bg=PURPLE_BG, fg=TEXT_COLOR, font=("Verdana", 11)).pack()

    hours_list   = [f"{i:02d}" for i in range(24)]
    minutes_list = [f"{i:02d}" for i in range(60)]
    dropdowns    = []
    time_slots   = []

    for i in range(3):
        fr = Frame(ts_win, bg=PURPLE_BG)
        fr.pack(pady=9)
        Label(fr, text=f"Slot {i+1}:", width=8, anchor='e',
              bg=PURPLE_BG, fg=TEXT_COLOR,
              font=("Verdana", 12)).pack(side=LEFT, padx=5)
        h_var = StringVar(value=f"{8+i:02d}")
        m_var = StringVar(value="00")
        om_h  = OptionMenu(fr, h_var, *hours_list)
        om_h.config(bg=ENTRY_BG, fg="white", width=4, font=("Verdana", 12))
        om_h.pack(side=LEFT, padx=3)
        Label(fr, text=":", bg=PURPLE_BG, fg=TEXT_COLOR,
              font=("Verdana", 14, "bold")).pack(side=LEFT)
        om_m  = OptionMenu(fr, m_var, *minutes_list)
        om_m.config(bg=ENTRY_BG, fg="white", width=4, font=("Verdana", 12))
        om_m.pack(side=LEFT, padx=3)
        dropdowns.append((h_var, m_var))

    val_lbl = Label(ts_win, text="", bg=PURPLE_BG, fg="#ff6666", font=("Verdana", 10))
    val_lbl.pack()
    res_lbl = Label(ts_win, text="", bg=PURPLE_BG, fg=ACCENT_COLOR,
                    font=("Verdana", 11, "bold"))
    res_lbl.pack(pady=6)

    def submit_times():
        nonlocal time_slots
        time_slots = []
        for i, (h_var, m_var) in enumerate(dropdowns):
            try:
                datetime.datetime.strptime(f"{h_var.get()}:{m_var.get()}", "%H:%M")
                time_slots.append(f"{h_var.get()}:{m_var.get()}")
            except ValueError:
                val_lbl.config(text=f"Invalid time in slot {i+1}.")
                return
        time_slots.sort()
        val_lbl.config(text="")
        res_lbl.config(
            text=f"✔  {time_slots[0]}   |   {time_slots[1]}   |   {time_slots[2]}"
        )

    Button(ts_win, text="CONFIRM SLOTS", command=submit_times,
           bg=PURPLE_LIGHT, fg=PURPLE_DARK,
           font=("Verdana", 12, "bold"),
           padx=20, pady=8, relief="flat").pack(pady=14)

    Button(ts_win, text="Close & Start",
           command=ts_win.destroy,
           bg=ACCENT_COLOR, fg=PURPLE_DARK,
           font=("Verdana", 12, "bold"),
           padx=20, pady=8, relief="flat").pack(pady=4)

    ts_win.grab_set()
    ts_win.wait_window()
    return time_slots if len(time_slots) == 3 else None


def wait_until(target_time_str):
    """Return seconds until the given HH:MM time today (or tomorrow if past)."""
    now    = datetime.datetime.now()
    target = datetime.datetime.combine(
        now.date(),
        datetime.datetime.strptime(target_time_str, "%H:%M").time()
    )
    if target < now:
        target += datetime.timedelta(days=1)
    return (target - now).total_seconds()


# ====================== MAIN ATTENDANCE FUNCTION ======================
def FillAttendance(text_to_speech_func, subject_window, tx, Notifica):
    sub = tx.get().strip()
    if not sub:
        text_to_speech_func("Please enter the subject name.")
        return

    time_slots = get_three_time_slots(subject_window)
    if not time_slots or len(time_slots) != 3:
        text_to_speech_func("Please select 3 time slots and click Confirm first.")
        return

    # ---- Load per-student models ----
    recognizers = load_per_student_recognizers()
    if not recognizers:
        text_to_speech_func("No trained models found. Please run Train Image first.")
        Notifica.config(text="No trained models found. Train first.", fg="#ff6666")
        return

    # ---- Load ArcFace embeddings (primary recognizer, more accurate than LBPH) ----
    embeddings = load_per_student_embeddings()
    if embeddings:
        print(f"Using ArcFace embeddings for {len(embeddings)} student(s).")
    else:
        print("No ArcFace embeddings found — falling back to LBPH only.")

    # ---- Load student registry ----
    try:
        student_df = pd.read_csv(STUDENTDETAIL_PATH)
    except Exception as e:
        text_to_speech_func("Could not read student details file.")
        print(f"CSV error: {e}")
        return

    text_to_speech_func(
        f"Starting attendance for {sub}. "
        f"Slots: {time_slots[0]}, {time_slots[1]}, {time_slots[2]}."
    )

    try:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        if not cam.isOpened():
            text_to_speech_func("Cannot open camera.")
            return

        win_name = f"Smart Attendance — {sub}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        text_to_speech_func("Camera opened. Monitoring active.")

        for slot_idx, slot in enumerate(time_slots):

            # ========== WAITING PHASE (phone detection active) ==========
            wait_secs  = wait_until(slot)
            start_wait = time.time()
            text_to_speech_func(f"Waiting for slot {slot_idx+1} at {slot}.")

            while time.time() - start_wait < wait_secs:
                ret, frame = cam.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # Detect faces so phone detection can use exclusion zones
                face_boxes = detect_faces_insightface(frame)
                frame, phone_confirmed = detect_phone(frame, face_boxes, sub)

                remaining = int(wait_secs - (time.time() - start_wait))
                cv2.putText(frame,
                    f"Slot {slot_idx+1}/3 | Waiting for {slot} | {remaining}s remaining",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

                if phone_confirmed:
                    cv2.putText(frame, "PHONE DETECTED!",
                                (20, 85), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cam.release()
                    cv2.destroyAllWindows()
                    return

            # ========== ATTENDANCE PHASE (60 seconds) ==========
            text_to_speech_func(f"Slot {slot_idx+1} started. Marking attendance now.")
            attendance   = {}   # { enrollment_id : name }
            att_start    = time.time()
            ATT_DURATION = 60

            while time.time() - att_start < ATT_DURATION:
                ret, frame = cam.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ---- Face detection + ArcFace embeddings (one InsightFace pass) ----
                faces_with_emb = get_faces_with_embeddings(frame)
                face_boxes = [box for box, _ in faces_with_emb]

                for (fx, fy, fw, fh), emb in faces_with_emb:
                    face_gray = gray[fy:fy+fh, fx:fx+fw]
                    if face_gray.size == 0:
                        continue

                    # ---- Try ArcFace embedding recognition first (handles beards) ----
                    eid = None; name = "Unknown"; label_conf = ""
                    if embeddings and emb is not None:
                        eid, name, sim = recognize_face_embedding(emb, embeddings, student_df)
                        if eid is not None:
                            label_conf = f"sim={sim:.2f}"
                        else:
                            # Fall back to LBPH for students without embedding files
                            eid, name, dist = recognize_face(face_gray, recognizers, student_df)
                            label_conf = f"d={dist:.0f}" if eid else f"d={dist:.0f}"
                    else:
                        eid, name, dist = recognize_face(face_gray, recognizers, student_df)
                        label_conf = f"d={dist:.0f}"

                    if eid is not None:
                        color = (0, 255, 0)
                        label = f"{name} | {label_conf}"
                        if eid not in attendance:
                            attendance[eid] = name
                    else:
                        color = (0, 0, 255)
                        label = f"Unknown | {label_conf}"

                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
                    cv2.putText(frame, label, (fx, fy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                # ---- Phone detection (uses face boxes as exclusion zones) ----
                frame, phone_confirmed = detect_phone(frame, face_boxes, sub)

                remaining = int(ATT_DURATION - (time.time() - att_start))
                cv2.putText(frame,
                    f"Slot {slot_idx+1}/3 | {slot} | {remaining}s | Present: {len(attendance)}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

                if phone_confirmed:
                    cv2.putText(frame, "PHONE DETECTED!",
                                (20, 85), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cam.release()
                    cv2.destroyAllWindows()
                    return

            # ---- Save this slot's attendance ----
            if attendance:
                date    = datetime.datetime.now().strftime("%Y-%m-%d")
                out_dir = os.path.join(ATTENDANCE_PATH, sub)
                os.makedirs(out_dir, exist_ok=True)
                csv_name = f"{sub}_{date}_{slot.replace(':', '-')}.csv"
                att_df   = pd.DataFrame(
                    [(eid, nm) for eid, nm in attendance.items()],
                    columns=["Enrollment", "Name"]
                )
                att_df.to_csv(os.path.join(out_dir, csv_name), index=False)
                status = f"Slot {slot_idx+1}: {len(attendance)} student(s) marked present."
            else:
                status = f"Slot {slot_idx+1}: No students detected."

            Notifica.config(text=status, bg=PURPLE_DARK, fg=ACCENT_COLOR)
            text_to_speech_func(status)

        text_to_speech_func(f"All 3 attendance slots completed for {sub}.")

    except Exception as e:
        print(f"Attendance error: {e}")
        import traceback
        traceback.print_exc()
        text_to_speech_func("An error occurred. Check the console.")
    finally:
        if 'cam' in locals() and cam.isOpened():
            cam.release()
        cv2.destroyAllWindows()


# ====================== SUBJECT CHOOSE WINDOW ======================
def subjectChoose(text_to_speech_func):
    subject_window = Tk()
    subject_window.title("Smart Attendance System")
    subject_window.geometry("620x420")
    subject_window.configure(background=PURPLE_BG)

    Label(subject_window, text="SMART CLASS ATTENDANCE",
          bg=PURPLE_BG, fg=ACCENT_COLOR,
          font=("Verdana", 20, "bold")).pack(pady=20)

    Label(subject_window, text="Enter Subject Name:",
          bg=PURPLE_BG, fg=TEXT_COLOR,
          font=("Verdana", 14)).pack()

    tx = Entry(subject_window, font=("Verdana", 18), bg=ENTRY_BG, fg=TEXT_COLOR,
               relief="flat", highlightbackground=PURPLE_LIGHT, highlightthickness=2)
    tx.pack(pady=10, ipadx=10, ipady=5)

    Notifica = Label(subject_window, text="", bg=PURPLE_DARK, fg=ACCENT_COLOR,
                     font=("Verdana", 11), height=3, relief="flat",
                     highlightbackground=PURPLE_LIGHT, highlightthickness=1)
    Notifica.pack(pady=15, fill="x", padx=20)

    btn_frame = Frame(subject_window, bg=PURPLE_BG)
    btn_frame.pack(pady=10)

    Button(btn_frame, text="▶  Start Attendance",
           command=lambda: FillAttendance(
               text_to_speech_func, subject_window, tx, Notifica),
           bg=PURPLE_LIGHT, fg=PURPLE_DARK,
           font=("Verdana", 13, "bold"),
           width=18, height=2, relief="flat").pack(side=LEFT, padx=10)

    Button(btn_frame, text="📂  View Records",
           command=lambda: (
               os.startfile(ATTENDANCE_PATH)
               if os.path.exists(ATTENDANCE_PATH) and tx.get().strip()
               else text_to_speech_func("Enter a subject name first.")
           ),
           bg=PURPLE_LIGHT, fg=PURPLE_DARK,
           font=("Verdana", 13, "bold"),
           width=18, height=2, relief="flat").pack(side=LEFT, padx=10)

    subject_window.mainloop()


# ====================== ENTRY POINT ======================
if __name__ == "__main__":
    subjectChoose(text_to_speech)