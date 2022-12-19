from keras.utils import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

# emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
# EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]


emotion_model_path = 'models/long_augmented_model.h5'                          # facial expression model 
EMOTIONS = ["angry" ,"disgust","scared", "happy", "neutral", "sad", "surprised"]


# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)



# starting video streaming
cv2.namedWindow('Facial Emotion Analysis')          # creates a window that can be used as a placeholder for images and trackbars
# camera = cv2.VideoCapture(r"c:/users/adel/desktop/videoplayback.mp4")                        # 0 refers to the webcam, can provide Path for a video
camera = cv2.VideoCapture(0)                        # 0 refers to the webcam, can provide Path for a video

while True:
    if not camera.isOpened():
        print('The camera is closed')
        continue
    success, frame = camera.read()
    if not success:
        print('Video capture failed')
        continue

    #reading the frame
    frame = imutils.resize(frame,width=800)         # Control the img aspect ratio by the width only
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8") # img for the Probabilities windows

    if len(faces) == 0:
        continue

    faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces

    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
    # the ROI for classification via the CNN
    roi = gray[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0               # (48, 48)
    roi = img_to_array(roi)                         # (48, 48, 1)
    roi = np.expand_dims(roi, axis=0)               # (1, 48, 48, 1)
    
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        w = int(prob * 300)
        var = i * 35
        cv2.rectangle(canvas, (7, var + 5), (w, var + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, var + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
    
    cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (51, 162, 255), 2)
    cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)




    cv2.imshow('Facial Emotion Analysis', frame)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(5) & 0xFF == ord('q'):          # read frames after at least 20 milesec, 0 milesec means forever
        break

camera.release()
cv2.destroyAllWindows()
