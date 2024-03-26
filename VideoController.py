import cv2

class Video:

    def __init__(self, source = 0):
        self.camera_ids = []
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

    def find_cameras(self, n_attempts = 5):
        index = 0
        self.camera_ids = []
        
        cap = cv2.VideoCapture()
        while n_attempts > 0:
            cap.open(index, cv2.CAP_DSHOW)

            if cap.isOpened():
                self.camera_ids.append(index)
                cap.release()
            else:
                n_attempts -= 1

            index += 1

        return self.camera_ids

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