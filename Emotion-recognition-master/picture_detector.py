from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

#file move to saved image folder
import os
import shutil

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
image_path = '../Images/'
saved_path = '../SavedImages/'
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

frame_name = ""
#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

     
max_emotion = "emotion"
max_prob = 0
frame_name = "emotion"
def Detector():
    image_names = os.listdir(image_path)
    for image_name in image_names:
        if(".jpg" or ".jpeg" or ".png" or ".gif" in image_name):
            frame_name = image_name
            


    # starting video streaming
    cv2.namedWindow('your_face')
    camera = cv2.VideoCapture()
    frame = cv2.imread(image_path + frame_name)

    print(frame_name)
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    print(len(faces))
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]


        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # draw the label + probability bar on the canvas
            # emoji_face = feelings_faces[np.argmax(preds)]

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
            (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                            (0, 0, 255), 2)


            if(i == 0):
                max_emotion =  emotion
                max_prob = prob

            else:
                if(prob > max_prob):
                    max_prob = prob
                    max_emotion = emotion
        
    else :
        
        max_emotion = "cannotdetectface"
        max_prob = 0


            
    print(" this is max emotion result ::::" , max_emotion , max_prob)
        # this_emotion = emotion
        # this_prob = prob
        # if(this_prob > Max_prob)
        # Max_prob = prob
        # Max_prob_emotion = emotion


    # cv2.imshow('your_face', frameClone)
    # cv2.imshow("Probabilities", canvas)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     print("q")

    camera.release()
    cv2.destroyAllWindows()
    file_move(frame_name)

    return max_emotion, max_prob
def file_move(frame_name):
    shutil.move(image_path + '\\' + frame_name , saved_path + '\\' + frame_name)


