import mediapipe as mp
import cv2, time, numpy as np
import os, sys

from matplotlib.pyplot import get_cmap
from mediapipe.framework.formats import landmark_pb2

from face_detection import Face_Detector
from card_detection import Card_Detector

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class Main_Detection:

    def __init__(self):
        self.category_mask = None
        self.blur_background = False
        self.options = None
        self.show_axis = True
        self.show_contour = True
        self.landmark_id = None
        self.image = None
        self.selected = (-11,-11)
        self.timestamp_ms = 0
        self.recorded_data = list()
        self.measures = []
        self.cmap = get_cmap("tab10")
        self.mask = None

        self.card_length_mm = 85.60 # 85 mm 

        self.enable_card_detection = False

        self.V_DIST_PX_old = None
        self.pixel_to_mm_factor = 1

        self.face_detector = Face_Detector()
        self.card_detector = Card_Detector()

    def process_image(self, image, stmp_ms=None):
        self.face_detector.process_image(image)

        if self.enable_card_detection:
            self.card_detector.process_image(image)

        self.image = image.copy()

    def normalized_to_pixel(self, p_normalized):

        image_height, image_width = self.image.shape[:2]
        x_px = min(p_normalized[0] * image_width -1, image_width  - 1)
        y_px = min(p_normalized[1] * image_height-1, image_height - 1)

        return np.round([x_px, y_px]).astype(int)

    def point_line_intersect(self, C, A, B):

        # Equation Line(A,B), its Perpendicular and intersection point.
        if (B[1] - A[1]) and (B[0] - A[0]): 
            m = (B[1] - A[1]) / (B[0] - A[0])
            
            n = A[1] - A[0]*m

            # Perpendicular
            new_m = -1.0/m
            new_n = C[1] - C[0]*new_m

            # intersect point:
            x = (new_n - n)/(m - new_m)
            y = x*m + n

        # Line(A,B) with m = inf.
        elif (B[1] - A[1]):
            x = A[0]
            y = C[1]
        # Line (A,B) with m = 0.
        elif (B[0] - A[0]):
            x = C[0]
            y = A[1]
        else:
            print("You passed the same point!")

        return np.array([x,y])

    def measure_user_exercises(self):
        if not self.face_detector.detection_result: return
        if not len(self.face_detector.detection_result.face_landmarks): return

        _, img_w = self.image.shape[:2]
        face_landmarks = self.face_detector.detection_result.face_landmarks[0]
        landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks])

     
        ### Getting vertical distance.
        VA_px = self.normalized_to_pixel(landmarks[10])
        VB_px = self.normalized_to_pixel(landmarks[152])
        self.V_DIST_PX = np.linalg.norm(VA_px-VB_px)

        ### Getting Vertical symetric line points.
        VA_px = self.normalized_to_pixel(landmarks[10])
        VB_px = self.normalized_to_pixel(landmarks[152])

        A = VA_px - img_w/20 * (VB_px-VA_px) # Extend line to edges.
        B = VB_px + img_w/20 * (VB_px-VA_px) # Extend line to edges.

        self.V_axis_points = np.round((A,B)).astype(int)

        ### Getting Horizontal eyes line points.
        HA_px = self.normalized_to_pixel(landmarks[33])
        HB_px = self.normalized_to_pixel(landmarks[263])

        A = HA_px - img_w/20 * (HB_px-HA_px) # Extend line to edges.
        B = HB_px + img_w/20 * (HB_px-HA_px) # Extend line to edges.

        self.H_axis_points = np.round((A,B)).astype(int)

        ######## CUSTOMIZABLE PART ######
        self.measures = []
        self.drawing_data = []
        if self.V_DIST_PX_old is None:
            self.V_DIST_PX_old = self.V_DIST_PX

        current_to_original_px  = self.V_DIST_PX_old / self.V_DIST_PX # Scale factor for re-scaling a px distance to the original px distance when the card was captured.
        
        for point_id, A_axis_id, B_axis_id in self.options:

            if point_id == -1: continue

            A_axis_px = self.normalized_to_pixel(landmarks[int(A_axis_id)])
            B_axis_px = self.normalized_to_pixel(landmarks[B_axis_id])

            P_px  = self.normalized_to_pixel(landmarks[point_id][:2]) # Mouth left edge.
            
            I = self.point_line_intersect(P_px, A_axis_px, B_axis_px) # Intersection point between Line(A, B) and a perpenicular line containing P.
            I_px = np.round(I).astype(int)

            dist_px = np.linalg.norm(P_px - I_px)

            original_dist_px        = dist_px * current_to_original_px
            dist_mm                 = original_dist_px * self.pixel_to_mm_factor

            self.drawing_data.append((P_px, I_px))
            self.measures.append(dist_mm)

        return self.measures, self.drawing_data

    def draw_measurements_on_image(self):

        if not self.face_detector.detection_result: return self.image

        face_landmarks_list = self.face_detector.detection_result.face_landmarks

        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Loop through the detected faces to visualize. (only one face)
        for face_landmarks in (face_landmarks_list):

            # Pre-processing landmarks list for removing unnecesary data.
            landmarks = landmark_pb2.NormalizedLandmarkList()
            landmarks.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])
            
            if self.show_contour:
                # Drawing facemesh contours.
                mp.solutions.drawing_utils.draw_landmarks(
                    image=self.image,
                    landmark_list=landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
                
            # Obtener índices únicos de los landmarks del contorno facial
            contour_points = np.array([[face_landmarks[start].x * self.image.shape[1], face_landmarks[start].y * self.image.shape[0]] 
                                        for start, _ in self.face_detector.SORTED_FACE_OVAL], dtype=np.int32).reshape((-1, 1, 2))

            cv2.drawContours(self.mask, [contour_points], -1, (255), thickness=cv2.FILLED)

            if self.show_axis:
                A, B = self.V_axis_points
                cv2.line(self.image, A, B, color=(0, 0, 255)  , thickness=1)

                A, B = self.H_axis_points
                cv2.line(self.image, A, B, color=(255, 255, 0), thickness=1)


            for i, (A,B) in enumerate(self.drawing_data):
                color = np.array(self.cmap(i)[:3])*255
                color[0], color[2] = color[2], color[0] # Change from RGB to BGR.

                cv2.line(self.image, A, B, color=color , thickness=2)

        return self.image

    def draw_face_mesh(self):
        if not self.face_detector.detection_result: return self.image
        if not len(self.face_detector.detection_result.face_landmarks): return self.image

        face_landmarks = self.face_detector.detection_result.face_landmarks[0]
        landmarks = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in face_landmarks]

        self.landmark_id = None

        selected_done = True if self.selected[0] == -11 else False  # If there is a point, check it.
        for id, landmark in enumerate(landmarks):
            P = self.normalized_to_pixel(landmark[:2])

            A, B = self.selected
            if not selected_done and A >= P[0]-1 and A <= P[0]+1 and B >= P[1]-1 and B <= P[1]+1:
                cv2.circle(self.image, P, 1, (0, 255, 0), -1)
                selected_done = True
                self.landmark_id = id
            else:
                # cv2.rectangle(annotated_image, P - 1, P + 1, (0, 0, 255), -1)
                cv2.circle(self.image, P, 1, (0, 0, 255), -1)


        return self.image

    def draw_blurred_background(self):

        if self.mask is not None and self.mask.shape == self.image.shape[:2]:
            img_copy = self.image.copy()

            self.image = cv2.boxFilter(self.image,-1, (11,11))
            self.image[self.mask>0] = img_copy[self.mask>0]

        return self.image

    def draw_card_detection(self):
        for detection in self.card_detector.detections:
            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
            cv2.rectangle(self.image, start_point, end_point, (0, 255, 0), 2)
            card_length_px = int(bbox.width)

            self.pixel_to_mm_factor = self.card_length_mm / card_length_px
            self.V_DIST_PX_old = self.V_DIST_PX
        
        return self.image


if __name__ == "__main__":

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = Main_Detection()

    detector.options = [[61, 33,263], [291,33,263]]

    detector.blur_background = False

    prev_time = time.time()
    
    while cap.isOpened(): 
        
        ret, frame = cap.read()
        if not ret: 
            print("[!]: Couldn't open the camera")
            break

        frame = cv2.flip(frame, 1)

        detector.process_image(frame)
        detector.measure_user_exercises()
        
        if detector.blur_background: 
            detector.draw_blurred_background()
        
        img = detector.draw_measurements_on_image()


        new_time = time.time()
        fps = round(1/(new_time-prev_time), 2)
        prev_time = new_time

        scale = 1
        img = cv2.resize(img, np.round((img.shape[1]*scale, img.shape[0]*scale)).astype(int), interpolation = cv2.INTER_CUBIC)
        cv2.putText(img, str(fps), (7, 35), cv2.FONT_HERSHEY_SIMPLEX , 1, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("img", img)

        # Write the frames into a video:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

