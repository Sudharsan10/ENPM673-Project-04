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
    new_intensities = I[new_points[:, 1].astype(
        int), new_points[:, 0].astype(int)].reshape(-1, 1)
    error = ROI[:, 3].reshape(-1, 1) - new_intensities
    dx = X_gradient[new_points[:, 1].astype(
        int), new_points[:, 0].astype(int)].reshape(-1, 1)
    dy = Y_gradient[new_points[:, 1].astype(
        int), new_points[:, 0].astype(int)].reshape(-1, 1)
    c1, c2 = dx*Template_points, dy*Template_points
    dIdW = np.hstack((c1[:, 0].reshape(-1, 1), c2[:, 0].reshape(-1, 1),
                      c1[:, 1].reshape(-1, 1), c2[:, 1].reshape(-1, 1), dx, dy))
    sig_dI_dwdp = np.matmul(dIdW.T, error)
    H_mat_inv = np.linalg.inv(np.matmul(dIdW.T, dIdW))
    delta_parameter = np.matmul(H_mat_inv, sig_dI_dwdp)
    updated_parameters = parameters.reshape(-1, 1) + delta_parameter
    delta_parameter = np.linalg.norm(delta_parameter)
    return updated_parameters.T, delta_parameter
# =============================================================================================================================================================================================================================== #
# Function Definition
# =============================================================================================================================================================================================================================== #


def lucas_kanade_algorithm(Template_data, I, parameters, start, end, threshold=0.035):
    newStart, newEnd = np.array([[start[0]], [start[1]], [1]]), np.array([
        [end[0]], [end[1]], [1]])
    X_gradient = cv.Sobel(I, cv.CV_64F, 1, 0, ksize=3)
    Y_gradient = cv.Sobel(I, cv.CV_64F, 0, 1, ksize=3)
    delta_parameter = 1
    while (delta_parameter > threshold):
        Warp_matrix = np.array([[1+parameters[0, 0], parameters[0, 2], parameters[0, 4]], [
                               parameters[0, 1], 1+parameters[0, 3], parameters[0, 5]]])
        parameters, delta_parameter = resolver(
            I, Warp_matrix, Template_data, X_gradient, Y_gradient, parameters)
        p1, p2 = np.matmul(Warp_matrix, newStart), np.matmul(
            Warp_matrix, newEnd)
    return Warp_matrix, parameters, p1, p2
# =============================================================================================================================================================================================================================== #
# Mouse callback function to Select the object
# =============================================================================================================================================================================================================================== #


def mark_the_object(event, x, y, flags, param):
    global start_point, end_point
    if event == cv.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        print("Start:", start_point)
    elif event == cv.EVENT_LBUTTONUP:
        end_point = (x, y)
        print("End:", end_point)
        cv.rectangle(template, (start_point[0], start_point[1]),
                     (end_point[0], end_point[1]), (0, 0, 255), 3)
        cv.imshow("Object Selection", template)
        cv.waitKey(2)

# =============================================================================================================================================================================================================================== #
# Object Detection Function
# =============================================================================================================================================================================================================================== #


def object_detection(name):
    global start_point, end_point, template
    # -----> Read the Key frame from the Video <----- #
    # Read the Video saved
    video = cv.VideoCapture('Videos/'+name+'.avi', 0)
    ret, key_frame = video.read()
    # Read the first key frame identify and mark the object
    template = key_frame.copy()
    # Converting the key frame to  Grayscale
    key_frame = cv.cvtColor(key_frame, cv.COLOR_BGR2GRAY)
    # Creating a named window to set callback function
    cv.namedWindow("Object Selection")
    # Set a Mouse callback to register the mouse action to extract the ROI
    cv.setMouseCallback("Object Selection", mark_the_object)
    # Show the first frame to select the object to be tracked
    cv.imshow("Object Selection", key_frame)
    cv.waitKey(0)
    # Get Height and Width of the Image frame
    cv.destroyAllWindows()
    # Get the height and width of the obj selected
    obj_height, obj_width = abs(
        start_point[1]-end_point[1]), abs(start_point[0]-end_point[0])
    # Region of Interest
    ROI_Data = np.zeros((obj_height * obj_width, 4))
    index = 0
    # -----> Extracting the Object information and Storing in ROI_Data <----- #
    for i in range(start_point[0], end_point[0]):
        for j in range(start_point[1], end_point[1]):
            ROI_Data[index] = [i, j, 1, key_frame[j, i]]
            index += 1
    count = 0
    template_mean = np.mean(key_frame)
    affine_parameters = np.zeros((1, 6))
    while True:
        ret, key_frame = video.read()
        if ret:
            key_frame_gray = cv.cvtColor(key_frame, cv.COLOR_BGR2GRAY)
            image_mean = np.mean(key_frame_gray)
            key_frame_gray_normalised = key_frame_gray.astype(float)
            key_frame_gray_normalised = key_frame_gray_normalised * \
                (template_mean/image_mean)
            W_mat, affine_parameters, st, en = lucas_kanade_algorithm(
                ROI_Data, key_frame_gray_normalised, affine_parameters, start_point, end_point)
            top_left = np.array([[start_point[0]], [start_point[1]], [1]])
            top_right = np.array([[end_point[0]], [start_point[1]], [1]])
            bottom_left = np.array([[start_point[0]], [end_point[1]], [1]])
            bottom_right = np.array([[end_point[0]], [end_point[1]], [1]])

            top_left = np.matmul(W_mat, top_left)
            top_right = np.matmul(W_mat, top_right)
            bottom_left = np.matmul(W_mat, bottom_left)
            bottom_right = np.matmul(W_mat, bottom_right)

            cv.line(key_frame, (top_left[0], top_left[1]),
                    (top_right[0], top_right[1]), (0, 0, 255), 2)
            cv.line(key_frame, (top_left[0], top_left[1]),
                    (bottom_left[0], bottom_left[1]), (0, 0, 255), 2)
            cv.line(key_frame, (bottom_right[0], bottom_right[1]),
                    (top_right[0], top_right[1]), (0, 0, 255), 2)
            cv.line(key_frame, (bottom_left[0], bottom_left[1]),
                    (bottom_right[0], bottom_right[1]), (0, 0, 255), 2)

            # cv.rectangle(key_frame, (st[0], st[1]), (en[0], en[1]), (0, 0, 255), 3)
            cv.imwrite('Output/'+name+'/' +
                       str(count).zfill(4)+'.jpg', key_frame)
            cv.imshow("Detection", key_frame)
            key = cv.waitKey(1)
            count += 1
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()
    # -----> Import Car Images & Write Video <----- #
    images = vw.import_images('Output/'+name)
    vw.video_writer(images, name+'_output', 23)


# =============================================================================================================================================================================================================================== #
# Run the program by uncommenting the appropriate lines below
# =============================================================================================================================================================================================================================== #
start_point, end_point = tuple(), tuple()

# Uncomment to run car detection
object_detection('car')
# object_detection('human')                                                                                                   # Uncomment to run Human detection
# object_detection('vase')                                                                                                    # Uncomment to run Vase detection
