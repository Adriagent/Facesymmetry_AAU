import cv2, sys, os
import mediapipe as mp
import time

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class Card_Detector:

    model_path = resource_path('models/efficientdet_lite0_32.tflite')

    def __init__(self):
        self.timestamp_ms = 0
        self.detections = []
        self.start_times = {}
        self.processing_complete = True  # Flag to track processing state

        Solution = mp.tasks.vision.ObjectDetector
        SolutionOptions = mp.tasks.vision.ObjectDetectorOptions

        options = SolutionOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            max_results=1,
            score_threshold=0.1,
            category_allowlist=["cell phone"],
            category_denylist=[],
            result_callback=self.result_callback
        )
        
        self.solution = Solution.create_from_options(options)

    def process_image(self, image, stmp_ms=None):
        self.timestamp_ms = stmp_ms if stmp_ms else self.timestamp_ms + 250
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        self.solution.detect_async(mp_image, self.timestamp_ms)

    def result_callback(self, result: mp.tasks.vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms):
        self.detections = result.detections
        self.processing_complete = True  # Set processing to True when done

    

if __name__ == "__main__":
    detector = Card_Detector()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        detector.process_image(frame)

        if detector.detections:
            # Draw the detections on the frame
            frame,_ = detector.draw_detections(frame)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)  # Introduce a short delay to limit frame processing rate

    cap.release()
    cv2.destroyAllWindows()
