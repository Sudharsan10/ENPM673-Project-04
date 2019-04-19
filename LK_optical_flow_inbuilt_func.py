# ======================================================================================================================================================================= #
#-------------> Project 04 <---------------#
# ======================================================================================================================================================================= #
# Course    :-> ENPM673 - Perception for Autonomous Robots
# Date      :-> 23 April 2019
# Authors   :-> Niket Shah(UID: 116345156), Siddhesh(UID: 116147286), Sudharsan(UID: 116298636)
# ======================================================================================================================================================================= #

# ======================================================================================================================================================================= #
# Import Section for Importing library
# ======================================================================================================================================================================= #
import time
import sys
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('Videos/car.avi')

# -----> <----- #
lk_params = dict(winSize = (20,20), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
_, frame = cap.read()
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)



# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x,y)
        point_selected = True
        old_points = np.array([[x,y]], dtype = np.float32)

cv.namedWindow('Frame')
cv.setMouseCallback('Frame', select_point)

point_selected = False
point = ()
old_points = np.array([[]])

while True:
    ret, frame = cap.read()

    if ret:

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if point_selected is True:
            
            cv.circle(frame, point, 30, (0, 0, 255), 2 )

            new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
            old_gray = gray_frame.copy()
            old_points = new_points

            x, y = new_points.ravel()
            cv.circle(frame, (x, y), 20, (255, 0, 0), -1)

        cv.imshow('Frame', frame)

        key = cv.waitKey(80)
        if key == 27: break
    else: break

cap.release()
cv.destroyAllWindows()
