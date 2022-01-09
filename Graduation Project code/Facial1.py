import face_recognition
import cv2
import numpy as np

cap = cv2.VideoCapture(r'Video test\1.mp4')

while True:
    ret, frame = cap.read()
    if ret:
        # image = frame# cv2.resize(frame, (300,300))
        image =  cv2.resize(frame, (800,600))
        
        # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        face_landmarks_list = face_recognition.face_landmarks(image)#, model='small')

        features_dict = dict()

        count=0
        for i in range(0,len(face_landmarks_list)):
            for feature in face_landmarks_list[0]:
                count+=1
                # print("FEATURE is ",feature)
                if feature in features_dict:
                    # append the new number to the existing array at this slot
                    #print("Appending...",face_landmarks_list[0][feature])
                    for j in range(0,len(face_landmarks_list[i][feature])):
                        features_dict[feature].append(face_landmarks_list[i][feature][j])
                else:
                    # create a new array in this slot
                    features_dict[feature] = face_landmarks_list[i][feature]

        print("Ran",count,"times")
        print('lLPAPLAPLAPAL')

        count=0
        for feature in features_dict:
            print(feature)
            count+=1
            for x in features_dict[feature]:
                #print('X is ',x)
                #print("Len of X is ",len(x))
                #if(len(x)==2):
                cv2.circle(image,x, 1, (0,255,0), 2)

        cv2.imshow('Output',image)
        if cv2.waitKey(1) == 13:
            break

cap.release()
cv2.destroyAllWindows()