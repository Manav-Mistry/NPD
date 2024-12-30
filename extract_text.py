import pandas as pd 
import cv2
import numpy as np
import easyocr
import os

reader = easyocr.Reader(['en'])

def apply_easyocr(image):
    extracted_text = reader.readtext(image)
    if len(extracted_text) > 0:
        # returns (text, score)    
        return (extracted_text[0][1], extracted_text[0][2])
    else:
        return (0, 0)


def getDesiredLocation(filename):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    folder_name = 'output_results'
    folder_path = os.path.join(current_file_directory, folder_name)
    desired_location = os.path.join(folder_path, filename)
    return desired_location


fileName = './output_results/output.h5'
df = pd.read_hdf(fileName)

df.to_csv(getDesiredLocation("ocr_results.csv"))

