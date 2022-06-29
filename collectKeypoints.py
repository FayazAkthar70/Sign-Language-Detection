import cv2 as cv
import holistic_tracking.holisticTrackingModule as htm
import make_folder as mkf
import os 
import numpy as np

detector = htm.holisticDetector()
vid = cv.VideoCapture(0)

for action in mkf.actions:
        for sequence in range(mkf.start_folder, mkf.start_folder+mkf.no_sequence):
            for frame_num in range(mkf.sequence_length):
                
                isTrue, img = vid.read()
                img = cv.flip(img, 1)
                img, results = detector.find_body(img)
                
                if frame_num == 0: 
                    cv.putText(img, 'STARTING COLLECTION', (120,200), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
                    cv.putText(img, f'Collecting frames for {action} Video Number {sequence}', (15,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    cv.imshow('OpenCV Feed', img)
                    cv.waitKey(1000)
                else: 
                    cv.putText(img, f'Collecting frames for {action} Video Number {sequence}', (15,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    cv.imshow('OpenCV Feed', img)
                
                keypoints = detector.extract_keypoints(results)
                npy_path = os.path.join(mkf.DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                
                if cv.waitKey(10) & 0xFF == ord("q"):   #press q to close video
                    break
                
vid.release()
cv.destroyAllWindows()