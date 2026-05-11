# ====================== trainImage.py ======================
# Each student gets their OWN separately trained LBPH model.
# Trainer_1.yml  = trained ONLY on student ID 1 images
# Trainer_2.yml  = trained ONLY on student ID 2 images
# They are NOT copies. Each model learns ONE student's face patterns.
# Combined Trainner.yml is also saved for any legacy code.
# =============================================================
import os
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_student_images(folder_path, enrollment_id):
    """
    Load all .jpg/.jpeg/.png images from a student folder.
    Returns (faces_list, ids_list).
    Images are converted to grayscale and resized to 200x200.
    """
    faces = []
    ids   = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            img      = Image.open(fpath).convert("L")   # grayscale
            arr      = np.array(img, dtype="uint8")
            arr      = cv2.resize(arr, (200, 200))
            faces.append(arr)
            ids.append(enrollment_id)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
    return faces, ids


def _compute_and_save_embeddings(student_dirs, trainimage_path, label_dir):
    """
    After LBPH training, compute one ArcFace embedding per student from their
    training images and save it as Embedding_<ID>.npy.

    WHY: LBPH fails to distinguish bearded/similar-looking students because it
    compares pixel-level texture patterns. ArcFace embeddings are 512-dimensional
    identity vectors trained on millions of faces — they correctly separate
    students even when they have similar beards, face shapes, or skin tones.

    The 200x200 grayscale face crops saved during registration are converted to
    112x112 BGR (the format InsightFace's recognition sub-model expects) and
    passed directly to the recognition model, bypassing the detection step.

    Returns the number of students for whom embeddings were saved (0 if
    InsightFace is not installed).
    """
    try:
        from insightface.app import FaceAnalysis
        import insightface as _if
        face_app = FaceAnalysis(
            name="buffalo_s",
            root=BASE_DIR,
            providers=["CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=-1, det_size=(112, 112))
        rec_model = face_app.models.get('recognition')
        if rec_model is None:
            print("InsightFace recognition model not found in buffalo_s — skipping embeddings")
            return 0
    except ImportError:
        print("InsightFace not installed — skipping ArcFace embedding computation")
        return 0
    except Exception as e:
        print(f"InsightFace init failed during embedding computation: {e}")
        return 0

    saved = 0
    for folder_name in sorted(student_dirs):
        folder_path = os.path.join(trainimage_path, folder_name)
        parts = folder_name.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            enrollment_id = int(parts[0])
        except ValueError:
            continue

        all_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not all_files:
            continue

        # Sample up to 300 images uniformly (keeps training fast even for
        # students with 2500 images).
        step = max(1, len(all_files) // 300)
        sampled = all_files[::step]

        feat_list = []
        for fname in sampled:
            fpath = os.path.join(folder_path, fname)
            try:
                gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                # Resize to 112x112 (InsightFace recognition model input size)
                img_112 = cv2.resize(gray, (112, 112))
                # Convert grayscale → BGR for InsightFace
                img_bgr = cv2.cvtColor(img_112, cv2.COLOR_GRAY2BGR)
                feat = rec_model.get_feat(img_bgr)
                if feat is not None and feat.size > 0:
                    feat_list.append(feat.flatten())
            except Exception as e:
                print(f"  Embedding error for {fname}: {e}")

        if not feat_list:
            print(f"  No embeddings computed for {folder_name}")
            continue

        # Average all sample embeddings and L2-normalize
        avg_emb = np.mean(feat_list, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb /= norm

        emb_path = os.path.join(label_dir, f"Embedding_{enrollment_id}.npy")
        np.save(emb_path, avg_emb)
        print(f"  Embedding_{enrollment_id}.npy saved "
              f"({len(feat_list)} of {len(all_files)} images used, folder: {folder_name})")
        saved += 1

    return saved


def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path,
               message, text_to_speech):
    """
    Train ONE individual LBPH model per student.

    What happens internally:
    - For student ID=1 (ABC):   reads all images in TrainingImage/1_ABC/
                                  trains a fresh LBPH recognizer on ONLY those images
                                  saves as TrainingImageLabel/Trainer_1.yml
    - For student ID=2 (XYZ):    reads all images in TrainingImage/2_XYZ/
                                  trains a COMPLETELY SEPARATE fresh LBPH recognizer
                                  saves as TrainingImageLabel/Trainer_2.yml
    - And so on for every student.

    Each Trainer_<ID>.yml file:
    - Was trained on a different student's images
    - Has NEVER seen any other student's face
    - Is NOT a copy of any other model

    This means during recognition, each model only matches its own student,
    greatly reducing the chance of false matches between students.
    """
    try:
        label_dir = os.path.dirname(trainimagelabel_path)
        os.makedirs(label_dir, exist_ok=True)

        if not os.path.exists(trainimage_path):
            msg = "TrainingImage folder not found. Register students first."
            message.configure(text=msg)
            text_to_speech(msg)
            return

        student_dirs = [
            d for d in os.listdir(trainimage_path)
            if os.path.isdir(os.path.join(trainimage_path, d))
        ]

        if not student_dirs:
            msg = "No student folders found. Please register students first."
            message.configure(text=msg)
            text_to_speech(msg)
            return

        total_students  = 0
        total_images    = 0
        all_faces_combined = []
        all_ids_combined   = []

        for folder_name in sorted(student_dirs):
            folder_path = os.path.join(trainimage_path, folder_name)
            parts = folder_name.split("_", 1)
            if len(parts) < 2:
                print(f"Skipping invalid folder name: {folder_name}")
                continue
            try:
                enrollment_id = int(parts[0])
            except ValueError:
                print(f"Cannot parse enrollment ID from: {folder_name}")
                continue

            faces, ids = _load_student_images(folder_path, enrollment_id)

            if len(faces) == 0:
                print(f"No images found for {folder_name}, skipping.")
                continue

            # ---- Train a completely fresh recognizer for THIS student only ----
            recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8
            )
            recognizer.train(faces, np.array(ids))

            model_path = os.path.join(label_dir, f"Trainer_{enrollment_id}.yml")
            recognizer.save(model_path)
            print(f"  Trainer_{enrollment_id}.yml saved — trained on {len(faces)} images of {folder_name}")

            total_students += 1
            total_images   += len(faces)

            # Collect for combined model
            all_faces_combined.extend(faces)
            all_ids_combined.extend(ids)

        if total_students == 0:
            msg = "Training failed: no valid student data found."
            message.configure(text=msg)
            text_to_speech(msg)
            return

        # ---- Save combined model (backward compatibility) ----
        if all_faces_combined:
            combined_rec = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8
            )
            combined_rec.train(all_faces_combined, np.array(all_ids_combined))
            combined_rec.save(trainimagelabel_path)
            print(f"  Combined Trainner.yml saved ({len(all_faces_combined)} total images)")

        # ---- Compute InsightFace ArcFace embeddings per student ----
        # These are saved as Embedding_<ID>.npy alongside the LBPH .yml files.
        # During attendance, the system uses embeddings FIRST (they are much
        # better at telling apart bearded/similar-looking students than LBPH)
        # and falls back to LBPH only if no embedding is available.
        emb_count = _compute_and_save_embeddings(
            student_dirs, trainimage_path, label_dir
        )
        if emb_count > 0:
            print(f"  ArcFace embeddings saved for {emb_count} student(s).")
        else:
            print("  ArcFace embeddings skipped (InsightFace not available).")

        msg = (f"Training Complete!\n"
               f"{total_students} students | {total_images} images trained.\n"
               f"ArcFace embeddings: {emb_count} student(s).\n"
               f"Each student has their own personal model file.")
        message.configure(text=msg)
        text_to_speech(f"Training complete. {total_students} students trained successfully.")
        print(msg)

    except Exception as e:
        err = f"Training Error: {str(e)}"
        message.configure(text=err)
        text_to_speech("Training failed. Check console for details.")
        print(err)
        import traceback
        traceback.print_exc()


# ---- Standalone test ----
if __name__ == "__main__":
    class FakeLabel:
        def configure(self, **kw):
            print("MSG:", kw.get("text", ""))

    def fake_tts(t):
        print("TTS:", t)

    TrainImage(
        haarcasecade_path="",
        trainimage_path=os.path.join(BASE_DIR, "TrainingImage"),
        trainimagelabel_path=os.path.join(BASE_DIR, "TrainingImageLabel", "Trainner.yml"),
        message=FakeLabel(),
        text_to_speech=fake_tts
    )