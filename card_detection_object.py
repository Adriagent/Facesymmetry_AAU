import cv2
import numpy as np
import mediapipe as mp
import time


class Detector:

    model_path = 'models/efficientdet_lite0_32.tflite'

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
        if not self.processing_complete:
            return
        
        self.processing_complete = False  # Set processing to False when starting
        self.timestamp_ms = stmp_ms if stmp_ms else self.timestamp_ms + 250
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Record the start time with the timestamp as the key
        self.start_times[self.timestamp_ms] = time.time()
        self.solution.detect_async(mp_image, self.timestamp_ms)

    # Callback function to handle the detection results
    def result_callback(self, result: mp.tasks.vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms):
        
        if not isinstance(timestamp_ms, int):
            timestamp_ms = timestamp_ms.value/1000

        
        # Retrieve the start time using the timestamp3
        start_time = self.start_times.pop(timestamp_ms, None)
        if start_time is not None:
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Callback processing time: {processing_time:.4f} seconds")

        self.detections = result.detections
        self.processing_complete = True  # Set processing to True when done

    def draw_detections(self, frame):
        for detection in self.detections:
            bbox = detection.bounding_box
            x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
            cropped_image = frame[y:y+h, x:x+w].copy()
            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

            # Draw label and confidence score
            label = detection.categories[0].category_name
            score = detection.categories[0].score
            label_text = f'{label}: {score:.2f}'
            label_position = (start_point[0], start_point[1] - 10)
            cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, cropped_image

if __name__ == "__main__":
    detector = Detector()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        detector.process_image(frame)

        if detector.detections:
            # Draw the detections on the frame
            frame, crop = detector.draw_detections(frame)
            cv2.imshow('Crop', crop)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)  # Introduce a short delay to limit frame processing rate

    cap.release()
    cv2.destroyAllWindows()
