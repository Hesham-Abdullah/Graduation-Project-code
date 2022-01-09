from keras.models import load_model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time


class Gen_keypoins():
    def __init__(self):
        self.model = load_model('checkpoints\model_1586626246.826244.h5')
        print(self.model.summary())

    def detect_points(self, face_img):
            me  = np.array(face_img)/255
            x_test = np.expand_dims(me, axis=0)
            x_test = np.expand_dims(x_test, axis=3)

            y_test = self.model.predict(x_test)
            label_points = (np.squeeze(y_test)*48)+48


            return label_points

    def get_points_main(self, img):

        # load haarcascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        dimensions = (96, 96)


        try:
            default_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(default_img, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
            # faces = face_cascade.detectMultiScale(gray_img, 4, 6)
            print(f'# faces: {len(faces)}')

        except:
            return []

        faces_img = np.copy(gray_img)

        plt.rcParams["axes.grid"] = False


        all_x_cords = []
        all_y_cords = []

        for i, (x,y,w,h) in enumerate(faces):

            h += 10
            w += 10
            x -= 5
            y -= 5

            try:
                just_face = cv2.resize(gray_img[y:y+h,x:x+w], dimensions)
            except:
                return []
            cv2.rectangle(faces_img,(x,y),(x+w,y+h),(255,0,0),1)

            scale_val_x = w/96
            scale_val_y = h/96

            label_point = self.detect_points(just_face)

            all_x_cords.append((label_point[::2]*scale_val_x)+x)
            all_y_cords.append((label_point[1::2]*scale_val_y)+y)



        final_points_list = []
        try:
            for ii in range(len(all_x_cords)):
                for a_x, a_y in zip(all_x_cords[ii], all_y_cords[ii]):
                    final_points_list.append([a_x, a_y])
        except:
            return final_points_list

        return final_points_list
