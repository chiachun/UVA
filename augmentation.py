#*********************************************************************#
# This piece of code is not used in run_uva.py.
# It's in ongoing development.
# Data augmentation can boost statistics and may
# improve clustering result of small samples.
#*********************************************************************#

import cv2
from skimage.transform import AffineTransform
from skimage.transform import warp

def augmentation(photoname,label):
    img = cv2.imread(photoname)
    labels = []
    zoom1s = [1.0]
    zoom2s = [1.0]
    rotations = [0,8]
    shears = [3,12]
    images = []
    flips = [False, True]
    for zoom1 in zoom1s:
        for zoom2 in zoom2s:
            for rotation in rotations:
                for shear in shears:
                    for flip in flips:
                        tform_augment = AffineTransform(scale=(1/zoom1, 1/zoom2), 
                                                        rotation=np.deg2rad(rotation), 
                                                        shear=np.deg2rad(shear))

                        img2 = warp(img, tform_augment)
                        if flip == True:
                            images.append(cv2.flip(img2,1))
                            labels.append(label)
                        else:
                            images.append(img2)
                            labels.append(label)
    return images,labels
