import cv2
import numpy as np
import mediapipe as mp
import time

class Detector:

    model_path = './selfie_multiclass_256x256.tflite'
    
    def __init__(self):
        self.timestamp_ms = 0
        self.confidence_masks = []
        self.start_times = {}
        self.processing_complete = True  # Flag to track processing state

        Solution = mp.tasks.vision.ImageSegmenter
        SolutionOptions = mp.tasks.vision.ImageSegmenterOptions

        options = SolutionOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
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
        
        self.solution.segment_async(mp_image, self.timestamp_ms)

    # Callback function to handle the segmented masks
    def result_callback(self, result, _, timestamp_ms):
        # Retrieve the start time using the timestamp
        start_time = self.start_times.pop(timestamp_ms, None)
        if start_time is not None:
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Callback processing time: {processing_time:.4f} seconds")
        
        self.confidence_masks = result.confidence_masks
        self.processing_complete = True  # Set processing to True when done

    def overlay_confidence_masks(self, frame, threshold=0.08):
        # Create an empty mask to hold the colored overlay
        if len(self.confidence_masks) != 6: 
            return frame

        colored_mask = np.zeros_like(frame)

        # Apply the threshold
        mask_np = self.confidence_masks[5].numpy_view()
        binary_mask = mask_np > threshold
        colored_mask[binary_mask] = (0, 255, 0)  # Green

        # Create a combined mask with transparency
        alpha = 0.25
        overlayed_frame = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)

        return overlayed_frame

if __name__ == "__main__":
    detector = Detector()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        detector.process_image(frame)

        if detector.confidence_masks:
            frame = detector.overlay_confidence_masks(frame) # Overlay the confidence masks on the frame

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
