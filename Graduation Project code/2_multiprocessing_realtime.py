# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:56:49 2020

@author: Workstation1
"""

import cv2
import numpy as np
from keras.models import load_model
from data import DataSet
import threading
import os
from multiprocessing import Process, Manager


def predictions(frames, model, manager_list, data, model_frames_num = 30):
    assert len(frames) == model_frames_num
    prediction = model.predict(np.expand_dims(frames, axis=0))
    pred = data.print_class_from_prediction(np.squeeze(prediction, axis=0))
    print(pred)
    manager_list.append(pred[0][0])


if __name__ == "__main__":
        
    sequence_list = []
    process_list = []
    manager = Manager()
    manager_list = manager.list()

    #capturing live video frames
    model_frames_num = 30
    realtime_frames = 60
    saved_model = 'data/checkpoints/lrcn-images.238-0.244.hdf5'
    model = load_model(saved_model)

    data = DataSet(seq_length=model_frames_num, class_limit=None)


    vid_path = os.path.join('Video test/2.mp4')           
    cap =cv2.VideoCapture(vid_path)
    dim = (300, 300)
    p = ''

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            full_fram = frame.copy()
            full_fram = cv2.resize(frame, (800,600), interpolation = cv2.INTER_AREA)
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            
            sequence_list.append(frame)
            
            if len(sequence_list) == realtime_frames:
                rescaledList = data.rescale_list(sequence_list, model_frames_num)
                pro = Process(target=predictions, args=(rescaledList, model, manager_list, data, model_frames_num,))
                pro.start()
                process_list.append(pro)
                # p = predictions(rescaledList)
                sequence_list = []
                rescaledList = []
            
            for idx, pp in enumerate(process_list):
                if (pp.is_alive == False):
                    pp.join()
                    print('\n[+]-------------------------')
                    print(pp[idx])
                    print('\n[-]-------------------------')
                    p += pp[idx]

            cv2.putText(full_fram, p, (250,550), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('sign view', full_fram)
            key = cv2.waitKey(1)
                
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()


