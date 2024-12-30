from ultralytics import YOLO
import cv2
import pandas as pd
from sort.sort import *
from util import get_car
import shutil
# import h5py
import easyocr


reader = easyocr.Reader(['en'])

def create_folder(folder_path):
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        # Empty the existing folder if it already exists
        print(f"Folder '{folder_path}' already exists. Emptying the folder.")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error while clearing folder: {e}")


def main():
    mot_tracker = Sort()

    # load models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('./best.pt')

    # load video
    cap = cv2.VideoCapture('./videos/input.mp4')

    vehicles = [2, 3, 5, 7]

    columns = ['car_id','npimage', 'xyxy','confidence_score']
    df = pd.DataFrame(columns=columns)

    # folder creation
    # Get the current file's directory
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    # Specify the folder name you want to create
    folder_name = 'output_results'
    # Create the folder in the current file's directory
    folder_path = os.path.join(current_file_directory, folder_name)
    create_folder(folder_path)

    # read frames
    frame_nmr = -1
    ret = True
    while ret :
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []

            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
                    
            # track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # gray scale
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                    extracted_text = reader.readtext(license_plate_crop_gray)
                    if len(extracted_text) > 0:
                        extracted_text_number = extracted_text[0][1]
                        extracted_text_score = extracted_text[0][2]
                    else:
                        extracted_text_number = 0
                        extracted_text_score = 0

                    new_data = {
                        'frame_nmr': frame_nmr,
                        'car_id': car_id,
                        'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                        'license_plate_bbox': [x1, y1, x2, y2],
                        'confidence_score': score,
                        'npimage': license_plate_crop_gray,
                        'license_number': extracted_text_number,
                        'license_number_score': extracted_text_score
                    }
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    desired_location = os.path.join(folder_path, "output.h5")
    df.to_hdf(desired_location, key="imagedata")            
                    

if __name__ == "__main__":
    main()