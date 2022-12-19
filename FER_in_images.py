from keras.utils import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys

# parameters for loading data and images
# https://github.com/opencv/opencv/tree/master/data/haarcascades
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'      # face detection model

# emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'                          # facial expression model 
# EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]


emotion_model_path = 'models/long_augmented_model.h5'                          # facial expression model 
EMOTIONS = ["angry" ,"disgust","scared", "happy", "neutral", "sad", "surprised"]

img_path = sys.argv[1]

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)                        # only take the path of the classification model
emotion_classifier = load_model(emotion_model_path, compile=False)

# check if the face detection model is loaded
assert not face_detection.empty(), "The face detection model has not been loaded"

# reading the frame
orig_frame = cv2.imread(img_path)           # If the image cannot be read, returns NULL --> (x, y , 3)
if orig_frame is None:                      # check if the image is read 
    assert False , f"The image cannot be read, PATH: ({img_path}) is INVALID" 

frame = cv2.imread(img_path,0)              # activate cv2.IMREAD_GRAYSCALE flag >> return gray image
faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

if len(faces) > 0:
    for (fX, fY, fW, fH) in faces:
        # (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]             # Cropping region of interest i.e. Face
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0               # (48, 48)
        roi = img_to_array(roi)                         # (48, 48, 1)
        roi = np.expand_dims(roi, axis=0)               # (1, 48, 48, 1)
        preds = emotion_classifier.predict(roi)[0]      # [[Probabilities of the classifications]]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin) ->	img
        cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (51, 162, 255), 2)

        # cv.rectangle(img, pt1, pt2, color, thickness, lineType, shift) ->	img
        cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
    
cv2.imshow('Facial emotion analysis', orig_frame)
cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)
if (cv2.waitKey(6000) & 0xFF == ord('q')):          # wait until 'q' key is pressed
    sys.exit("Thanks")
cv2.destroyAllWindows()                             # to close the window and de-allocate any associated memory usage

