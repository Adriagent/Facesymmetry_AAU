import mediapipe as mp
import cv2, time, numpy as np
import os, sys

from matplotlib.pyplot import get_cmap
from mediapipe.framework.formats import landmark_pb2

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class face_detector:

    landmarker_model_path = resource_path("./models/face_landmarker.task")

    SORTED_FACE_OVAL = [(10, 338), (338, 297), (297, 332), (332, 284),
                        (284, 251), (251, 389), (389, 356), (356, 454),
                        (454, 323), (323, 361), (361, 288), (288, 397),
                        (397, 365), (365, 379), (379, 378), (378, 400),
                        (400, 377), (377, 152), (152, 148), (148, 176),
                        (176, 149), (149, 150), (150, 136), (136, 172),
                        (172, 58), (58, 132), (132, 93), (93, 234),
                        (234, 127), (127, 162), (162, 21), (21, 54),
                        (54, 103), (103, 67), (67, 109), (109, 10)]

    def __init__(self):
        self.detection_result = None
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

        self.V_DIST_PX_old = None
        self.pixel_to_mm_factor = 1

        self.init_face_detector()

    def init_face_detector(self):
        BaseOptions             = mp.tasks.BaseOptions
        FaceLandmarker          = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions   = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode       = mp.tasks.vision.RunningMode

        face_options = FaceLandmarkerOptions(
                        base_options    = BaseOptions(model_asset_path=self.landmarker_model_path),
                        running_mode    = VisionRunningMode.LIVE_STREAM,
                        num_faces       = 1, 
                        result_callback = self.landmarker_result_callback)
        
        self.landmarker = FaceLandmarker.create_from_options(face_options)

    def process_image(self, image, stmp_ms=None):
        if stmp_ms:
            self.timestamp_ms = stmp_ms 
        else:  
            self.timestamp_ms+=1000  
        
        self.image = image.copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        self.landmarker.detect_async(mp_image, self.timestamp_ms)

    def landmarker_result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.detection_result = result
    
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
        if not self.detection_result: return
        if not len(self.detection_result.face_landmarks): return

        img_h, img_w = self.image.shape[:2]
        face_landmarks = self.detection_result.face_landmarks[0]
        landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks])

        self.measures = dict()
        self.drawing_data = dict()

        ### Getting vertical distance.
        VA = landmarks[10]
        VB = landmarks[152]
        VA_px = self.normalized_to_pixel(VA)
        VB_px = self.normalized_to_pixel(VB)
        V_DIST_PX = np.linalg.norm(VA_px-VB_px)

        ### Getting Vertical symetric line points.
        VA_px = self.normalized_to_pixel(landmarks[10])
        VB_px = self.normalized_to_pixel(landmarks[152])

        A = VA_px - img_w/20 * (VB_px-VA_px) # Extend line to edges.
        B = VB_px + img_w/20 * (VB_px-VA_px) # Extend line to edges.

        self.drawing_data["V_axis_points"] = np.round((A,B)).astype(int)

        ### Getting Horizontal eyes line points.
        HA_px = self.normalized_to_pixel(landmarks[33])
        HB_px = self.normalized_to_pixel(landmarks[263])

        A = HA_px - img_w/20 * (HB_px-HA_px) # Extend line to edges.
        B = HB_px + img_w/20 * (HB_px-HA_px) # Extend line to edges.

        self.drawing_data["H_axis_points"] = np.round((A,B)).astype(int)

        ### Drawing left mouth edge distance.
        C_px  = self.normalized_to_pixel(landmarks[61][:2])     # Mouth left edge.
        
        P = self.point_line_intersect(C_px, VA_px, VB_px) # Intersection point between Line(VA, VB) and a perpenicular line containing C.
        P_px = np.round(P).astype(int)

        dist = np.linalg.norm(C_px - P_px)
        dist_norm = dist / V_DIST_PX * 100     # Distance normalized to be scale invariant.

        self.drawing_data["L_mouth_V_dist"] = (C_px, P_px)
        self.measures["L_mouth_V_dist"] = dist_norm

        ### Drawing right mouth edge distance to vertical axis.
        C_px  = self.normalized_to_pixel(landmarks[291][:2])    # Mouth right edge.
        
        P = self.point_line_intersect(C_px, VA_px, VB_px) # Intersection point between Line(VA, VB) and a perpenicular line containing C.
        P_px = np.round(P).astype(int)

        dist = np.linalg.norm(C_px - P_px)
        dist_norm = dist / V_DIST_PX * 100     # Distance normalized to be scale invariant.

        self.drawing_data["R_mouth_V_dist"] = (C_px, P_px)
        self.measures["R_mouth_V_dist"] = dist_norm

        ### Drawing left motuth edge distance to horizontal axis.
        C_px  = self.normalized_to_pixel(landmarks[61][:2])     # Mouth left edge.
  
        P = self.point_line_intersect(C_px, HA_px, HB_px) # Intersection point between Line(VA, VB) and a perpenicular line containing C.
        P_px = np.round(P).astype(int)

        dist = np.linalg.norm(C_px - P_px)
        dist_norm = dist / V_DIST_PX * 100     # Distance normalized to be scale invariant.

        self.drawing_data["L_mouth_H_dist"] = (C_px, P_px)
        self.measures["L_mouth_H_dist"] = dist_norm

        ### Drawing right motuth edge distance to horizontal axis.
        C_px  = self.normalized_to_pixel(landmarks[291][:2])    # Mouth left edge.
        
        P = self.point_line_intersect(C_px, HA_px, HB_px) # Intersection point between Line(VA, VB) and a perpenicular line containing C.
        P_px = np.round(P).astype(int)

        dist = np.linalg.norm(C_px - P_px)
        dist_norm = dist / V_DIST_PX * 100     # Distance normalized to be scale invariant.

        self.drawing_data["R_mouth_H_dist"] = (C_px, P_px)
        self.measures["R_mouth_H_dist"] = dist_norm

    def draw_measurements_on_image(self):

        if not self.detection_result: return self.image

        face_landmarks_list = self.detection_result.face_landmarks

        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Loop through the detected faces to visualize. (only one face)
        for face_landmarks in (face_landmarks_list):

            # Pre-processing landmarks list for removing unnecesary data.
            landmarks = landmark_pb2.NormalizedLandmarkList()
            landmarks.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])

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
                                        for start, _ in self.SORTED_FACE_OVAL], dtype=np.int32).reshape((-1, 1, 2))

            cv2.drawContours(self.mask, [contour_points], -1, (255), thickness=cv2.FILLED)


            for key in self.drawing_data:
                A, B = self.drawing_data[key]

                if key == "V_axis_points":
                    cv2.line(self.image, A, B, color=(0, 0, 255) , thickness=1)
                elif key == "H_axis_points":
                    cv2.line(self.image, A, B, color=(255, 255, 0) , thickness=1)
                else:
                    cv2.line(self.image, A, B, color=(255, 0, 255) , thickness=2)


        return self.image

    def draw_face_mesh(self):
        if not self.detection_result: return self.image
        if not len(self.detection_result.face_landmarks): return self.image

        face_landmarks = self.detection_result.face_landmarks[0]
        landmarks = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in face_landmarks]

        for landmark in landmarks:
            P = self.normalized_to_pixel(landmark[:2])

            # cv2.rectangle(annotated_image, P - 1, P + 1, (0, 0, 255), -1)
            cv2.circle(self.image, P, 1, (0, 0, 255), -1)


        return self.image

    def measure_user_exercises_2(self):
        if not self.detection_result: return
        if not len(self.detection_result.face_landmarks): return

        _, img_w = self.image.shape[:2]
        face_landmarks = self.detection_result.face_landmarks[0]
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

    def draw_measurements_on_image_2(self):

        if not self.detection_result: return self.image

        face_landmarks_list = self.detection_result.face_landmarks

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
                                        for start, _ in self.SORTED_FACE_OVAL], dtype=np.int32).reshape((-1, 1, 2))

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

    def draw_face_mesh_2(self):
        if not self.detection_result: return self.image
        if not len(self.detection_result.face_landmarks): return self.image

        face_landmarks = self.detection_result.face_landmarks[0]
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


if __name__ == "__main__":

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = face_detector()

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

        # Write the frames into a video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

