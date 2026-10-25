import cv2
import threading
import queue
from ultralytics import YOLO

class WebcamStream:
    """Threaded webcam reader to prevent video stutter and frame drops."""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.q = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def stop(self):
        self.stopped = True
        self.cap.release()

def main():
    model = YOLO("yolo11n.pt")

    stream = WebcamStream(src=0).start()
    print("\n==========================================")
    print(" 🚀 REAL-TIME YOLO SYSTEM STARTED")
    print(" Press 'Q' on the video window to exit.")
    print("==========================================\n")

    while not stream.stopped:
        frame = stream.read()
        if frame is None:
            continue

        results = model.predict(frame, conf=0.35, verbose=False)[0]
        annotated_frame = results.plot()

        cv2.imshow("Optimized Real-Time YOLO", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()