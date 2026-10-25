import os
import cv2
import numpy as np
import threading
import queue
import logging

# Suppress unnecessary TensorFlow / Abseil warnings in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from ultralytics import YOLO
from deepface import DeepFace

# --- CONFIGURATION ---
DATABASE_DIR = "registered_persons"
FACE_MODEL = "ArcFace"  # Ultra-fast and accurate face representation model
THRESHOLD = 0.50        # Cosine distance cutoff (lower = stricter match)
MATCH_INTERVAL = 4      # Run face verification every N frames for high FPS

os.makedirs(DATABASE_DIR, exist_ok=True)

# Load lightweight detectors
print("⚡ Initializing YOLOv8 and Haar Cascade detectors...")
yolo_model = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global database cache and thread communication queues
known_embeddings = {}
task_queue = queue.Queue()
results_cache = {}  # {person_id: name}
is_running = True

def load_database():
    """Load cached facial embeddings into memory."""
    global known_embeddings
    known_embeddings = {}
    if not os.path.exists(DATABASE_DIR):
        return

    for file in os.listdir(DATABASE_DIR):
        if file.endswith(".npy"):
            person_name = os.path.splitext(file)[0]
            try:
                known_embeddings[person_name] = np.load(os.path.join(DATABASE_DIR, file))
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")

    print(f"✅ Loaded {len(known_embeddings)} registered profile(s) from database.")

def extract_face_crop(image_bgr):
    """Isolates the exact facial area out of a person crop."""
    if image_bgr is None or image_bgr.size == 0:
        return None
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) == 0:
        return None
    
    # Pick the largest face in the crop frame
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return image_bgr[y:y+h, x:x+w]

def register_person_fast(frame, person_name: str):
    """Instantly extracts facial features and updates database."""
    face_crop = extract_face_crop(frame)
    
    if face_crop is None:
        print("\n❌ Registration Failed: No clear face detected. Ensure your face is visible and well-lit.")
        return False

    try:
        embedding_objs = DeepFace.represent(
            img_path=face_crop, 
            model_name=FACE_MODEL, 
            enforce_detection=False
        )
        embedding = embedding_objs[0]["embedding"]

        # Save to disk
        save_path = os.path.join(DATABASE_DIR, f"{person_name}.npy")
        np.save(save_path, embedding)
        print(f"\n🎉 Successfully registered '{person_name}'!")
        
        # Reload database cache
        load_database()
        return True
    except Exception as e:
        print(f"\n❌ Error during registration: {e}")
        return False

def verify_face_worker():
    """Background worker thread to run face matching without dropping main video FPS."""
    global results_cache
    while is_running:
        try:
            person_id, face_crop = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if not known_embeddings or face_crop is None or face_crop.size == 0:
            results_cache[person_id] = "Unknown"
            task_queue.task_done()
            continue

        try:
            crop_objs = DeepFace.represent(
                img_path=face_crop, 
                model_name=FACE_MODEL, 
                enforce_detection=False
            )
            target_embedding = crop_objs[0]["embedding"]

            best_match = "Unknown"
            min_distance = float("inf")

            for name, saved_embedding in known_embeddings.items():
                # Cosine distance computation
                dot_product = np.dot(target_embedding, saved_embedding)
                norms = np.linalg.norm(target_embedding) * np.linalg.norm(saved_embedding)
                dist = 1.0 - (dot_product / max(norms, 1e-6))

                if dist < min_distance and dist < THRESHOLD:
                    min_distance = dist
                    best_match = name

            results_cache[person_id] = best_match
        except Exception:
            results_cache[person_id] = "Unknown"

        task_queue.task_done()

def run_pipeline():
    """Main execution loop."""
    global is_running
    load_database()

    # Start background face verification worker thread
    worker_thread = threading.Thread(target=verify_face_worker, daemon=True)
    worker_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera source.")
        is_running = False
        return

    # Set camera resolution (640x480 for higher FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n=============================================")
    print(" 🎥 REAL-TIME SYSTEM READY")
    print(" -------------------------------------------")
    print("  Press 'R' -> Register person in current frame")
    print("  Press 'Q' -> Quit Application")
    print("=============================================\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        frame_count += 1
        height, width, _ = frame.shape

        # Run YOLO detection for persons (Class 0)
        results = yolo_model(frame, verbose=False)[0]

        current_frame_ids = []

        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            if yolo_model.names[cls_id] == "person":
                xyxy = list(map(int, box.xyxy[0].tolist()))
                x1, y1, x2, y2 = xyxy
                
                # Boundary safety checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                person_id = f"person_{idx}"
                current_frame_ids.append(person_id)

                # Queue face matching task every N frames to keep stream real-time
                if frame_count % MATCH_INTERVAL == 0:
                    person_crop = frame[y1:y2, x1:x2]
                    face_crop = extract_face_crop(person_crop)
                    
                    if face_crop is not None and task_queue.qsize() < 2:
                        task_queue.put((person_id, face_crop))

                # Retrieve recognized name or default label
                name_label = results_cache.get(person_id, "Detecting...")
                
                # Pick box color: Green for known person, Red for unknown
                color = (0, 255, 0) if name_label not in ["Unknown", "Detecting..."] else (0, 0, 255)

                # Draw bounding box and display name
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(name_label) * 15, y1), color, -1)
                cv2.putText(frame, name_label, (x1 + 5, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show live output window
        cv2.imshow("Optimized YOLO + Face System", frame)

        key = cv2.waitKey(1) & 0xFF

        # --- KEYBOARD CONTROLS ---
        if key in [ord('r'), ord('R')]:
            # Pause UI overlay for registration text prompt
            cv2.putText(frame, "REGISTRATION MODE: Check CMD Prompt", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Optimized YOLO + Face System", frame)
            cv2.waitKey(100)

            name = input("\n👉 Enter Name for Person Registration: ").strip()
            if name:
                register_person_fast(frame, name)

        elif key in [ord('q'), ord('Q')]:
            print("\nShutting down system...")
            break

    is_running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()