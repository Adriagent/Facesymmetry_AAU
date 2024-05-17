import cv2, numpy as np
from pygrabber.dshow_graph import FilterGraph

class Video:

    def __init__(self, source = 0):
        self.available_cameras = {}
        self.source = source
        self.cap = None

    def open_video(self, source=None):
        if source is not None:
            self.source = source

        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"[!]: Cannot open camera: {self.source}")
            return False

        return True 

    def find_cameras(self):

        devices = FilterGraph().get_input_devices()

        self.available_cameras = {}

        for device_index, device_name in enumerate(devices):
            self.available_cameras[device_index] = device_name

        return self.available_cameras

    def get_frame(self):
        if self.cap is None:
            print("[W]: No source was selected!")
            return False, None

        ret, img = self.cap.read()

        if not ret:
            print("[W]: No image was obtained!")
        else:
            img = cv2.flip(img, 1)

        return ret, img
    
    def set_frame_index(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

        
if __name__ == "__main__":

    cameras = Video()

    print(cameras.find_cameras())