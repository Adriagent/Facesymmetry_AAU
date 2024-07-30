import mediapipe as mp
import os, sys

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class Face_Detector:

    model_path = resource_path("./models/face_landmarker.task")

    SORTED_FACE_OVAL = [(10, 338), (338, 297),  (297, 332), (332, 284),
                        (284, 251), (251, 389), (389, 356), (356, 454),
                        (454, 323), (323, 361), (361, 288), (288, 397),
                        (397, 365), (365, 379), (379, 378), (378, 400),
                        (400, 377), (377, 152), (152, 148), (148, 176),
                        (176, 149), (149, 150), (150, 136), (136, 172),
                        (172, 58),  (58, 132),  (132, 93),  (93, 234),
                        (234, 127), (127, 162), (162, 21),  (21, 54),
                        (54, 103),  (103, 67),  (67, 109),  (109, 10)]

    def __init__(self):
        self.detection_result = None
        self.image = None
        self.timestamp_ms = 0

        Solution = mp.tasks.vision.FaceLandmarker
        SolutionOptions = mp.tasks.vision.FaceLandmarkerOptions

        options = SolutionOptions(
            base_options    = mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode    = mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback = self.result_callback)
        
        self.solution = Solution.create_from_options(options)

    def process_image(self, image, stmp_ms=None):
        if stmp_ms:
            self.timestamp_ms = stmp_ms 
        else:  
            self.timestamp_ms+=1000  
        
        self.image = image.copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        self.solution.detect_async(mp_image, self.timestamp_ms)

    def result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.detection_result = result


