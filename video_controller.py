import cv2
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
            exit(f"[!]: Cannot open camera: {self.source}")

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
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

        
if __name__ == "__main__":

    cameras = Video()

    print(cameras.find_cameras())