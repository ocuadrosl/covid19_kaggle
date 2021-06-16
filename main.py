
import numpy as np
import pandas as pd
import os
import cv2
import pydicom
import matplotlib.pyplot as plt
from pydicom.errors import InvalidDicomError
from pydicom.pixel_data_handlers.util import apply_voi_lut
import sys



def changeDepth(dicom, depth):

    cvType = cv2.CV_8U if depth == 8 else cv2.CV_16U
    npType = np.uint8 if depth == 8 else np.uint16

    data = apply_voi_lut(dicom.pixel_array, dicom)

    # depending on this value, X-ray may look inverted - fix that:
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # depth = np.dtype(npType).itemsize * 8
    nBits = (2**depth)-1

    normalizedImg = np.zeros(np.shape(data), dtype=np.dtype(npType))

    cvType = cv2.CV_8U if depth == 8 else cv2.CV_16U
    normalizedImg = cv2.normalize(data, dst=normalizedImg, alpha=0, beta=nBits, norm_type=cv2.NORM_MINMAX, dtype=cvType)

    #print(type(normalizedImg[0, 0]))
    return normalizedImg


if __name__ == '__main__':

    # setup args from command line
    datasetPath =  sys.argv[1]
    datasetWrite =  sys.argv[2]
    datasetTarget = sys.argv[3]
    depth = int(sys.argv[4])

    # for histogram equalization
    clahe = cv2.createCLAHE(tileGridSize=(8, 8))

    for dirname, _, filenames in os.walk(os.path.join(datasetPath, datasetTarget)):
        for filename in filenames:
            #print(os.path.join(dirname, filename))

            try:
                dicom = pydicom.dcmread(os.path.join(dirname, filename))

                # change image depth
                image = changeDepth(dicom, depth)

                # histogram equalization
                eqImage = clahe.apply(image)

                # write equalized images
                fileNameTIFF = os.path.splitext(filename)[0]+".tiff"
                if not cv2.imwrite(os.path.join(datasetWrite, datasetTarget, fileNameTIFF), eqImage):
                    raise ('Could not write image')

                # plt.imshow(eqImage, cmap=plt.cm.gray)
                # plt.show()
            except (InvalidDicomError, TypeError, RuntimeError) as raises:
                print(filename)
                print(raises)