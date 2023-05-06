import cv2 as cv
import holistic_tracking.holisticTrackingModule as htm
import make_folder as mkf
import numpy as np
import joblib

# Load the model from the file
model = joblib.load('signLanguageModel.pkl')

detector = htm.holisticDetector()
vid = cv.VideoCapture(0)
input_frames = []     #input frames from camera consisting of 30 frames each     
predictions = []

while True:
    isTrue, img = vid.read()
    img = cv.flip(img, 1)
    img, results = detector.find_body(img)
    keypoints = detector.extract_keypoints(results)
    input_frames.append(keypoints)
    
    if len(input_frames) == 30:
        cv.waitKey(1000)
        res = model.predict(np.expand_dims(input_frames, axis=0))[0]
        print(mkf.actions[np.argmax(res)])
        predictions.append(mkf.actions[np.argmax(res)])
        input_frames = []
    if predictions:
        prediction_text = " ".join(predictions[-1])
    else:
        prediction_text = " "
    cv.putText(img, f'The word you said is {prediction_text}', (10,200), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
    cv.imshow('OpenCV Feed', img)

    if cv.waitKey(10) & 0xFF == ord("q"):   #press q to close video
        break

vid.release()
cv.destroyAllWindows()