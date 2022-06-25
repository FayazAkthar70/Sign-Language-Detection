from unittest import result
import cv2 as cv
import time
import holisticTrackingModule as htm

current_time = 0
prev_time = 0
detector = htm.holisticDetector()
vid = cv.VideoCapture(0)

while True:
    isTrue, img = vid.read()
    img, result = detector.find_body(img)
    print(result)
    current_time = time.time()
    fps = str(int(1/(current_time-prev_time)))
    prev_time = current_time
    cv.putText(img, fps, (10,70), cv.FONT_HERSHEY_COMPLEX, 2,(0,0,0), 3)
    
    cv.imshow('video',img)
    if cv.waitKey(10) & 0xFF == ord("q"):   #press q to close video
        break
    
vid.release()
cv.destroyAllWindows()