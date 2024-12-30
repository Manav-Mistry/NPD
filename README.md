
# Real Time Number Plate Detection in video using YOLOv8 

Detects number plate for car, motorbike, bus, and truck. Stores the detection in a csv file, and then use that data to extract number plate text. Finally, it will create an output video which displays most accurate number plate text along with the bounding boxes.


## Demo

Input Video:
![](https://gitea.fuzzyapps.in/FuzzyML/number-plate-ditaction/src/branch/master/input-gif.gif)

Output Video:
![](https://gitea.fuzzyapps.in/FuzzyML/number-plate-ditaction/src/branch/master/out-gif.gif)


## Models

* Object Detection : Yolov8n from __*ultralytics*__

* License Plate Detection : best.pt which is trained on Roboflow dataset. 
    
    [Dataset Link](https://universe.roboflow.com/matheus-santos-almeida/car_license_plate/dataset/1)
## Additional Resources

* Used [github repository](https://github.com/abewley/sort) for real time __*Object Tracking*__  
## Method

### The process is divided mainly into five parts.

* [__*i*__] Vehicle Detection
* [__*ii*__] Number Plate Detection
* [__*iii*__] Associating Number Plate with all the detected vehicles in the frame
* [__*iv*__] Saving all the detections data into a csv file
* [__*v*__] Generate an output video with visualization from the data

## Installation

Clone this repository with  

```bash
  git clone https://gitea.fuzzyapps.in/FuzzyML/number-plate-ditaction.git
```

Also clone below repository for object tracking

```bash
  git clone https://github.com/abewley/sort.git
```


## Run Locally
* create and activate virtual environment 

* Install packages under the requirements.txt file 

```bash
  pip install -r requirements.txt
```
* In the videos folder, replace the input video with the video you want detections 

* change iou_batch method in sort.py file under Sort folder
```bash
  def iou_batch(bb_test, bb_gt):

   bb_gt = np.expand_dims(bb_gt, 0)
   bb_test = np.expand_dims(bb_test, 1)
   if bb_test.size == 0 or bb_gt.size == 0:
      return np.zeros((0,))
   xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
   yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
   xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
   yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
   w = np.maximum(0., xx2 - xx1)
   h = np.maximum(0., yy2 - yy1)
   wh = w * h
   o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
      + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
   return(o)  
```

* Run the main.py file
```bash
  python -u main.py
```

* After that, run extract_text.py file

```bash
  python -u extract_text.py
```

* Finally, run visualization.py file to generate output video

```bash
  python -u visualization.py
```
This will create an output video in the videos folder.
