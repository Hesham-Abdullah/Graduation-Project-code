from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
from generate_keypoints import Gen_keypoins

def main():
    
    gk = Gen_keypoins()
    # input video file path
    input_file = r'D:\Mahmoud Nada\GP Code\Video test\1.mp4'


    # output file path
    output_filename = 'testVideo_out.avi'  


    cap = cv2.VideoCapture(input_file)
    ret, frame = cap.read()
    height, width, channel = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))


    frame_no = 0
    while cap.isOpened():

        a = time.time()
        
        frame_no += 1
        ret, frame = cap.read()
        # if frame_no > 75*30:
        #     break
        if True:# frame_no in range(60*30, 75*30):
            points = gk.get_points_main(frame)
            print(points)
            try:
                overlay = frame #.copy()
            except Exception as e:
                print(e)
                break

            for point in points:

                cv2.circle(frame, tuple(point), 3, (255, 255, 255), -1)
                # cv2.line(frame, last_point, tuple(point), (0,0,255), thickness=1)
                # cv2.putText(overlay, str(i), tuple(point), 1, 1, (255, 255, 255))

            if len(points) != 0:
                o_line_points = [[12,13], [13,11], [11,14], [14,12], [12,10], [11,10], [10,3], [12,5], [11,3], [10,5], [10,4], [10,2], [5,1], [1,4], [2,0], [0,3], [5,9], [9,8], [8,4], [2,6], [6,7], [7,3]]
                num_face = len(points)//15

                for i in range(num_face):
                    line_points = np.array(o_line_points) + (15*(i))

                    the_color = (189, 195, 199)

                    for ii in line_points:
                        cv2.line(overlay, tuple(points[ii[0]]), tuple(points[ii[1]]), the_color, thickness=1)


            opacity = 0.3
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

            out.write(frame)
            cv2.imshow('frame',frame)
            # cv2.imshow('overlay',overlay)
            b = time.time()
            # print(str((b-a)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()