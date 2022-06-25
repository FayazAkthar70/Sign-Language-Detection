import cv2 as cv
import mediapipe as mp
import time



class holisticDetector():
    def __init__(self, mode=False, complexity=1, smooth=True, is_segment=False, smooth_segment=True, refine_face=False, min_confi=0.5, min_conf=0.5):
        self.mode = mode
        self.complexity = complexity 
        self.smooth = smooth
        self.is_segment = is_segment
        self.smooth_segment = smooth_segment
        self.refine_face = refine_face
        self.min_confi = min_confi
        self.min_conf = min_conf
        
        self.mpHolistic = mp.solutions.mediapipe.python.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(mode, complexity, smooth, is_segment, smooth_segment, refine_face, min_confi, min_conf)
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
    
    def find_body(self, img):
        imgRGB = cv.cvtColor(img, code=cv.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)

        # if self.results.:
        #     for hand in self.results.:
        #         self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img, self.results
            
    # def find_position(self, img, draw=True, hand_no=0):
    #     hand_landmarks = []
    #     if self.results.multi_hand_landmarks:
    #         my_hand = self.results.multi_hand_landmarks[hand_no]
    #         for id, landmark in enumerate(my_hand.landmark):
    #             h, w, c = img.shape
    #             cx, cy = int(w*landmark.x), int(h*landmark.y)
    #             hand_landmarks.append([id,cx,cy])
    #     return hand_landmarks
    def draw_landmarks(self, img):
        if self.results.face_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.face_landmarks, self.mpHolistic.FACEMESH_CONTOURS,
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=2, circle_radius=1),
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=2, circle_radius=1))
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS,
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=2, circle_radius=1),
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=2, circle_radius=1))
        if self.results.left_hand_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS,
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=2, circle_radius=1),
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=2, circle_radius=1))
        if self.results.right_hand_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS,
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=2, circle_radius=1),
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=2, circle_radius=1))
        return img
 

