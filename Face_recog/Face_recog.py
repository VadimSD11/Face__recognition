import face_recognition
import os
import datetime
import cv2
import numpy as np
from deepface import DeepFace





path = 'Faces'
images = []
classNames = []
myList = os.listdir(path)



for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

cap = cv2.VideoCapture(0)
recoring=True
frame_size = (int(cap.get(3)),int(cap.get(4)))
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
current_datetime = datetime.datetime.now()
video_filename = f"video_{current_datetime.strftime('%Y%m%d_%H%M')}.mp4"

# Open a new VideoWriter with the unique filename
out = cv2.VideoWriter(video_filename, fourcc, 20, frame_size)

face_recognition_enabled = True 
mood_enabled=False

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    if face_recognition_enabled:

       for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Emotion and Age Detection
            if mood_enabled:
            # Emotion and Age Detection
                try:
                    face_img = img[y1:y2, x1:x2]
                    results = DeepFace.analyze(face_img, actions=['emotion', 'age'], enforce_detection=False)
    
                    # Ensure that a face was detected before accessing results
                    if 'emotion' in results[0]:
                        emotions = results[0]['emotion']
                        age = results[0]['age']
                        # Find the emotion with the highest percentage
                        max_emotion = max(emotions, key=emotions.get)
                        max_emotion_percentage = emotions[max_emotion]

                        cv2.putText(img, f'Age: {age}', (x1 + 6, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(img, f'Emotion: {max_emotion} ({max_emotion_percentage:.2f}%)', (x1 + 6, y2 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        print("Face not detected in the specified region.")
                except Exception as e:
                    print(f"Error analyzing face: {e}")


        else:
            name = 'Unknown'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if mood_enabled:
                try:
                    face_img = img[y1:y2, x1:x2]
                    results = DeepFace.analyze(face_img, actions=['emotion', 'age'], enforce_detection=False)
    
                    # Ensure that a face was detected before accessing results
                    if 'emotion' in results[0]:
                        emotions = results[0]['emotion']
                        age = results[0]['age']
                        # Find the emotion with the highest percentage
                        max_emotion = max(emotions, key=emotions.get)
                        max_emotion_percentage = emotions[max_emotion]

                        cv2.putText(img, f'Age: {age}', (x1 + 6, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(img, f'Emotion: {max_emotion} ({max_emotion_percentage:.2f}%)', (x1 + 6, y2 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        print("Face not detected in the specified region.")
                except Exception as e:
                    print(f"Error analyzing face: {e}")


    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, timestamp, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    out.write(img)
   
        

            # # Emotion and Age Detection



    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        face_recognition_enabled = not face_recognition_enabled
    elif key==ord('t'):        
            mood_enabled=not mood_enabled
            if face_recognition_enabled ==False:
                mood_enabled==False

cv2.destroyAllWindows()
out.release()
cap.release()
