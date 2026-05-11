# ====================== takeImage.py (YuNet ONLY - No Haar, No Box in Saved Images) ======================
import csv
import os
import cv2
import numpy as np
import time


def initialize_yunet(yunet_model_path):
    """Initialize YuNet face detector for image capture"""
    try:
        if os.path.exists(yunet_model_path):
            yunet = cv2.FaceDetectorYN.create(
                model=yunet_model_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            print("YuNet model loaded successfully for image capture")
            return yunet
        else:
            print(f"YuNet model not found at {yunet_model_path}")
            return None
    except Exception as e:
        print(f"Error initializing YuNet: {e}")
        return None


def detect_face_yunet(frame_bgr, yunet_detector):
    """
    Detect best face using YuNet. Returns (x, y, w, h, confidence) or None.
    """
    if yunet_detector is None:
        return None
    try:
        h, w = frame_bgr.shape[:2]
        yunet_detector.setInputSize((w, h))
        _, faces = yunet_detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[-1])
        x, y, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        conf = float(best[-1])
        return (x, y, fw, fh, conf)
    except Exception as e:
        print(f"YuNet error: {e}")
        return None


def TakeImage(l1, l2, frontal_default_path, frontal_alt_path, frontal_alt2_path,
              profile_path, upperbody_path, yunet_path, trainimage_path,
              message, err_screen, text_to_speech):

    if (l1 == "") and (l2 == ""):
        text_to_speech('Please Enter your Enrollment Number and Name.')
        return
    elif l1 == '':
        text_to_speech('Please Enter your Enrollment Number.')
        return
    elif l2 == "":
        text_to_speech('Please Enter your Name.')
        return

    try:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        yunet_detector = initialize_yunet(yunet_path)
        if yunet_detector is None:
            text_to_speech("YuNet model not found. Cannot capture images.")
            cam.release()
            return

        Enrollment = l1.strip()
        Name = l2.strip()

        directory = Enrollment + "_" + Name
        path = os.path.join(trainimage_path, directory)
        os.makedirs(path, exist_ok=True)

        angle_counters = {'front': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        target_per_angle = 500
        total_target = target_per_angle * 5
        angle_sequence = ['front', 'left', 'right', 'up', 'down']
        angle_index = 0
        current_angle = 'front'

        guidance_messages = {
            'front': "LOOK STRAIGHT AT CAMERA",
            'left':  "TURN FACE FULLY LEFT",
            'right': "TURN FACE FULLY RIGHT",
            'up':    "TILT FACE UPWARD",
            'down':  "TILT FACE DOWNWARD"
        }

        sampleNum = 0
        last_capture_time = 0
        capture_delay = 0.05  # 20 fps capture rate

        text_to_speech("Please look straight at the camera.")


        while True:
            ret, img = cam.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)

            # ---- IMPORTANT: keep a CLEAN copy for saving (NO drawings on it) ----
            clean_img = img.copy()

            detection = detect_face_yunet(img, yunet_detector)

            # ---------- DISPLAY overlay on img (NOT on clean_img) ----------
            cv2.putText(img, f"INSTRUCTION: {guidance_messages[current_angle]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            y_off = 60
            for ang in angle_sequence:
                done = angle_counters[ang] >= target_per_angle
                marker = "> " if ang == current_angle else "  "
                color = (0, 255, 0) if done else ((255, 255, 0) if ang == current_angle else (200, 200, 200))
                cv2.putText(img, f"{marker}{ang.upper()}: {angle_counters[ang]}/{target_per_angle}",
                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y_off += 28

            total_done = sum(angle_counters.values())
            cv2.putText(img, f"TOTAL: {total_done}/{total_target}",
                        (10, y_off + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            if detection is not None:
                x, y_f, w, h, conf = detection

                # Draw box ON DISPLAY FRAME ONLY (img), never on clean_img
                cv2.rectangle(img, (x, y_f), (x + w, y_f + h), (0, 255, 0), 2)
                cv2.putText(img, f"YuNet {conf:.2f}", (x, y_f - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                current_time = time.time()
                if current_time - last_capture_time >= capture_delay:
                    if angle_counters[current_angle] < target_per_angle:
                        # Crop from CLEAN image (no box drawn on it)
                        margin = int(min(w, h) * 0.1)
                        y1 = max(0, y_f - margin)
                        y2 = min(clean_img.shape[0], y_f + h + margin)
                        x1 = max(0, x - margin)
                        x2 = min(clean_img.shape[1], x + w + margin)

                        face_crop = clean_img[y1:y2, x1:x2]
                        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        face_resized = cv2.resize(face_gray, (200, 200))

                        sampleNum += 1
                        angle_counters[current_angle] += 1

                        filename = f"{Name}_{Enrollment}_{current_angle}_{angle_counters[current_angle]}.jpg"
                        cv2.imwrite(os.path.join(path, filename), face_resized)
                        last_capture_time = current_time

                        cv2.putText(img, "CAPTURED!", (x, y_f + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(img, "NO FACE DETECTED - Move closer or adjust lighting",
                            (10, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.imshow("Student Registration - Face Capture", img)

            # Advance angle when target reached
            if angle_counters[current_angle] >= target_per_angle:
                if angle_index < len(angle_sequence) - 1:
                    angle_index += 1
                    current_angle = angle_sequence[angle_index]
                    msg_map = {
                        'left': "Front done. Turn face fully LEFT.",
                        'right': "Left done. Turn face fully RIGHT.",
                        'up': "Right done. Tilt face UPWARD.",
                        'down': "Up done. Tilt face DOWNWARD."
                    }
                    text_to_speech(msg_map.get(current_angle, ""))

            if all(c >= target_per_angle for c in angle_counters.values()):
                cv2.putText(img, "ALL IMAGES CAPTURED! Press any key.",
                            (60, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Student Registration - Face Capture", img)
                cv2.waitKey(4000)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save student record
        os.makedirs("StudentDetails", exist_ok=True)
        file_exists = os.path.isfile("StudentDetails/studentdetails.csv")
        with open("StudentDetails/studentdetails.csv", "a+", newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=",")
            if not file_exists or os.path.getsize("StudentDetails/studentdetails.csv") == 0:
                writer.writerow(["Enrollment", "Name"])
            writer.writerow([Enrollment, Name])

        res = f"Images Saved for {Name} ({Enrollment}). Total: {sampleNum} images."
        message.configure(text=res)
        text_to_speech(f"Image capture complete. {sampleNum} images saved for {Name}.")

    except FileExistsError:
        text_to_speech("Student Data already exists")
        if err_screen:
            err_screen()
    except Exception as e:
        print(f"TakeImage Error: {e}")
        text_to_speech("An error occurred during image capture.")
