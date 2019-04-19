# =============================================================================================================================================================================================================================== #
#-------------> Project 04 <---------------#
# =============================================================================================================================================================================================================================== #
# Course    :-> ENPM673 - Perception for Autonomous Robots
# Date      :-> 23 April 2019
# Authors   :-> Niket Shah(UID: 116345156), Siddhesh(UID: 116147286), Sudharsan(UID: 116298636)
# =============================================================================================================================================================================================================================== #

# =============================================================================================================================================================================================================================== #
# Import Section for Importing library
# =============================================================================================================================================================================================================================== #
import time
import sys
import numpy as np
import cv2 as cv
import video_writer as vw

# =============================================================================================================================================================================================================================== #
# Function Definition
# =============================================================================================================================================================================================================================== #
def resolver(I, W_mat, ROI, X_gradient, Y_gradient, parameters):
    Template_points = ROI[:, 0:2]
    new_points = np.matmul(W_mat, ROI[:, 0:3].T).T
    new_intensities = I[new_points[:, 1].astype(int), new_points[:, 0].astype(int)].reshape(-1, 1)
    error = ROI[:, 3].reshape(-1, 1) - new_intensities
    dx = X_gradient[new_points[:,1].astype(int),new_points[:,0].astype(int)].reshape(-1, 1)
    dy = Y_gradient[new_points[:,1].astype(int),new_points[:,0].astype(int)].reshape(-1, 1)    
    c1,c2 = dx*Template_points, dy*Template_points
    dIdW = np.hstack((c1[:, 0].reshape(-1,1), c2[:, 0].reshape(-1,1), c1[:, 1].reshape(-1,1), c2[:, 1].reshape(-1,1), dx, dy))
    sig_dI_dwdp = np.matmul(dIdW.T, error)
    H_mat_inv = np.linalg.inv(np.matmul(dIdW.T, dIdW))
    delta_parameter = np.matmul(H_mat_inv, sig_dI_dwdp)
    updated_parameters = parameters.reshape(-1, 1) + delta_parameter
    delta_parameter = np.linalg.norm(delta_parameter)
    return updated_parameters.T, delta_parameter
# =============================================================================================================================================================================================================================== #
# Function Definition
# =============================================================================================================================================================================================================================== #
def lucas_kanade_algorithm(Template_data, I, parameters, start, end, threshold = 0.035):
    newStart, newEnd = np.array([ [start[0]], [start[1]], [1] ]), np.array([ [end[0]], [end[1]], [1] ])
    X_gradient = cv.Sobel(I, cv.CV_64F, 1, 0, ksize=3)
    Y_gradient = cv.Sobel(I, cv.CV_64F, 0, 1, ksize=3)
    delta_parameter = 1
    while (delta_parameter > threshold ):
        Warp_matrix = np.array([ [1+parameters[0,0], parameters[0,2], parameters[0,4]], [parameters[0,1], 1+parameters[0,3], parameters[0,5]] ])
        parameters, delta_parameter = resolver(I, Warp_matrix, Template_data, X_gradient, Y_gradient, parameters)
        p1, p2 = np.matmul(Warp_matrix, newStart), np.matmul(Warp_matrix, newEnd)
    return Warp_matrix, parameters, p1, p2
# =============================================================================================================================================================================================================================== #
# Mouse callback function to Select the object
# =============================================================================================================================================================================================================================== #
def mark_the_object(event, x, y, flags, param):
    global start_point, end_point
    if event == cv.EVENT_LBUTTONDOWN:
        start_point = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        end_point = (x, y)

# =============================================================================================================================================================================================================================== #
# Import the Video
# =============================================================================================================================================================================================================================== #
start_point, end_point = tuple(), tuple()
# -----> Read the Key frame from the Video <----- #
video = cv.VideoCapture('Videos/car.avi', 0 )                                                                           # Read the Video saved 
ret, key_frame = video.read()                                                                                           # Read the first key frame identify and mark the object
key_frame = cv.cvtColor(key_frame, cv.COLOR_BGR2GRAY)                                                                   # Converting the key frame to  Grayscale
cv.namedWindow("Object Selection")                                                                                      # Creating a named window to set callback function
cv.setMouseCallback("Object Selection", mark_the_object)                                                                # Set a Mouse callback to register the mouse action to extract the ROI
cv.imshow("Object Selection", key_frame)                                                                                # Show the first frame to select the object to be tracked 
cv.waitKey(0)
cv.destroyAllWindows()
frame_height, frame_width = key_frame.shape                                                                             # Get Height and Width of the Image frame
obj_height, obj_width = abs(start_point[1]-end_point[1]), abs(start_point[0]-end_point[0])                              # Get the height and width of the obj selected
ROI_Data = np.zeros((obj_height * obj_width, 4))                                                                        # Region of Interest 
index = 0
# -----> Extracting the Object information and Storing in ROI_Data <----- #
for i in range(start_point[0], end_point[0]):
    for j in range(start_point[1], end_point[1]):
        ROI_Data[index] = [i, j, 1, key_frame[j, i]]
        index += 1
count = 0
template_mean = np.mean(key_frame)
affine_parameters = np.zeros((1,6))
while True:
    ret, key_frame = video.read()
    if ret:
        key_frame_gray = cv.cvtColor(key_frame, cv.COLOR_BGR2GRAY)
        image_mean = np.mean(key_frame_gray)
        key_frame_gray_normalised = key_frame_gray.astype(float)
        key_frame_gray_normalised = key_frame_gray_normalised * (template_mean/image_mean)
        W_mat, affine_parameters, st, en = lucas_kanade_algorithm(ROI_Data, key_frame_gray_normalised, affine_parameters, start_point, end_point)
        cv.rectangle(key_frame, (st[0], st[1]), (en[0], en[1]), (0, 0, 255), 3)
        cv.imwrite("Output/car/"+str(count)+".jpg", key_frame)
        cv.imshow("Detection", key_frame)
        key = cv.waitKey(1)
        count += 1
        if key == ord('q'): break
    else: break
video.release()
cv.destroyAllWindows()

# -----> Import Car Images & Write Video <----- #
car = vw.import_images('Output/car')
vw.video_writer(car, 'car_output', 23)

# # -----> Import human Images & Write Video <----- #
# human = vw.import_images('Output/human')
# vw.video_writer(human, 'human_output', 23)

# # -----> Import Vase Images & Write Video <----- #
# vase = vw.import_images('Output/vase')
# vw.video_writer(vase, 'vase_output', 23)
