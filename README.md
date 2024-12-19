# Evaluation Analysis
This repository contains the scripts for following fatures:-
- Evaluating the performance of 2D Object Detection models. 
- Plotting the Precision and Reclcation curves.
- Comparing the AP values of all objects classes.
- OpenCV visualizer for visualizing the detections and ground truths of dataset.

## TODO
- [ ] Add Preview section.
- [ ] Add funtionality for more object detection frameworks. 
- [ ] Add requirement.txt file. 

## Getting started
1. Setup the environment
```
# Clone the repo
git clone --recursive https://github.com/gaurav00700/EvaluationAnalysis-2DObjectDetection.git

#Create the Python environment
conda create -n eval_img python=3.10 -y
conda activate eval_img

#Install using requirements
pip install -r requirements.txt

```

2. Run visualization script (example) [Kitti360](scripts/Img_EvaluateAnalysis.py)
```
python scripts/Img_EvaluateAnalysis.py --checkpoint <add checkpoint path --model_name yolo2d --image_dir <image dataset dir> --ann_dir <annotation .jsons dataset dir>
```

## Folder structure

```
├── assests
│ ├── ...
├── data
│ ├── input     (to the scripts)
│ ├── output    (from the scripts)  
├── lib
│ ├── evaluation_img.py
│ ├── utils_eval.py
│ ├── visualization.py 
├── scripts
│ ├── Img_EvaluateAnalysis.py
│ ├── ...
├── README.md
```