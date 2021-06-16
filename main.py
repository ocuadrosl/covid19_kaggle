
import numpy as np
import pandas as pd
import os
import cv2
import pydicom
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.preprocessing import normalize

#import open



def changeDepth(dicom, uType, fix_monochrome=True, gamma=1.0):

    data = apply_voi_lut(dicom.pixel_array, dicom)

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    depth = np.dtype(uType).itemsize * 8
    nBits = (2**depth)-1

    normalizedImg = np.zeros(np.shape(data), dtype=np.dtype(uType))

    cvType = cv2.CV_8U if depth == 8 else cv2.CV_16U
    normalizedImg = cv2.normalize(data, dst=normalizedImg, alpha=0, beta=nBits, norm_type=cv2.NORM_MINMAX, dtype=cvType)

    #print(type(normalizedImg[0, 0]))
    return normalizedImg


if __name__ == '__main__':

    clahe = cv2.createCLAHE(tileGridSize=(8, 8))
    for dirname, _, filenames in os.walk('/home/oscar/data/siim-covid19-detection/original/train'):
        for filename in filenames:
            #print(os.path.join(dirname, filename))

            try:
                dicom = pydicom.dcmread(os.path.join(dirname, filename))
                image = changeDepth(dicom, np.uint8)

                # histogram equalization
                eqImage = clahe.apply(image)
                # write equalized images
                fileNameTIFF = os.path.splitext(filename)[0]+".tiff"
                cv2.imwrite("/home/oscar/data/siim-covid19-detection/uint8/train/"+fileNameTIFF, eqImage)
                # plt.imshow(eqImage, cmap=plt.cm.gray)
                # plt.show()
            except:
                print(filename)
                print("Not a DCM file")


