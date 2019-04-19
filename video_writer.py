# ======================================================================================================================================================================= #
#-------------> Project 01 <---------------#
# ======================================================================================================================================================================= #
# Course    :-> ENPM673 - Perception for Autonomous Robots
# Date      :-> 23 April 2019
# Authors   :-> Niket Shah(UID: 116345156), Siddhesh(UID: 116147286), Sudharsan(UID: 116298636)
# ======================================================================================================================================================================= #

# ======================================================================================================================================================================= #
# Import Section for Importing library
# ======================================================================================================================================================================= #
import time, sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import math, glob
from scipy import stats

# ======================================================================================================================================================================= #
# Function for Importing Images from coressponding folders
# ======================================================================================================================================================================= #
def import_images(foldername: str)-> dict:
    images = np.array([cv.imread(img) for img in glob.glob(foldername+"/*.jpg")])
    return images

# ======================================================================================================================================================================= #
# Function to write video to the folders
# ======================================================================================================================================================================= #
def video_writer(data: np.array, name: str, frames_per_sec: int)-> None:
    width = data[0].shape[1]
    height = data[0].shape[0]

    print('width:',width,'height:',height)
    print(data[0].shape)

    video = cv.VideoWriter('Videos/'+name+'.avi', cv.VideoWriter_fourcc(*'XVID'), frames_per_sec, (width, height))

    for key_frame in data:
        video.write(key_frame)
        
    video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':    
    # -----> Import Car Images & Write Video <----- #
    car = import_images('Data/car')
    video_writer(car,'car', 23)

    # -----> Import human Images & Write Video <----- #
    human = import_images('Data/human')
    video_writer(human, 'human', 23)

    # -----> Import Vase Images & Write Video <----- #
    vase = import_images('Data/vase')
    video_writer(vase, 'vase', 23)

    print('Done..!')
