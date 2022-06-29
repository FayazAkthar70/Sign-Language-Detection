import cv2 as cv
import mediapipe as mp
import numpy as np



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
        
        
    
    def find_body(self, img, draw=True):
        imgRGB = cv.cvtColor(img, code=cv.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)
        if draw:
            self.draw_landmarks(img)
        return img, self.results
    
    def extract_pose(self, img):
        pose = []
        if self.results.pose_landmarks:
            for lnd in self.results.pose_landmarks.landmark:
                pose+= [lnd.x,lnd.y, lnd.z, lnd.visibility]
            return np.array(pose)
        else:
            return np.zeros(shape=132)   # 33 landmarks * 4(x,y and visibility). Dscarded z-axis as it is not used to determine depth as of now. ref: mediapipe documentation
    
    def extract_face(self, img):
        face = []
        if self.results.face_landmarks:
            for lnd in self.results.face_landmarks.landmark:
                face+= [lnd.x,lnd.y,lnd.z]
            return np.array(face)   
        else:
            return np.zeros(shape=1404) #468 landmarks * 3(x, y and z)
    
    def extract_left_hand(self, img):
        handl = []
        
        if self.results.left_hand_landmarks:
            for lnd in self.results.left_hand_landmarks.landmark:
                handl+= [lnd.x, lnd.y, lnd.z]
            return np.array(handl)    
        else:
            return np.zeros(shape=63)   # 21 landmarks * 3

    def extract_right_hand(self, img):
        handr = []
        if self.results.right_hand_landmarks:
            for  lnd in self.results.right_hand_landmarks.landmark:
                handr+= [lnd.x, lnd.y, lnd.z]
            return np.array(handr)
        else:
            return np.zeros(shape=63)   # 21 landmarks * 3
    
    def extract_keypoints(self, img):
        face = self.extract_face(img)
        pose = self.extract_pose(img)
        handl = self.extract_left_hand(img)
        handr = self.extract_right_hand(img)
        return np.concatenate([face, pose, handl, handr])
        
    def draw_landmarks(self, img):
        if self.results.face_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.face_landmarks, self.mpHolistic.FACEMESH_CONTOURS,
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=1, circle_radius=1),
                                       self.mpDraw.DrawingSpec((0), thickness=1, circle_radius=1))
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS,
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=1, circle_radius=1),
                                       self.mpDraw.DrawingSpec((0), thickness=1, circle_radius=1))
        if self.results.left_hand_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS,
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=1, circle_radius=1),
                                       self.mpDraw.DrawingSpec((0), thickness=1, circle_radius=1))
        if self.results.right_hand_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS,
                                       self.mpDraw.DrawingSpec((155,15,155), thickness=1, circle_radius=1),
                                       self.mpDraw.DrawingSpec((0), thickness=1, circle_radius=1))
 

