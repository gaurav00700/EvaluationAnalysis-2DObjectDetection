# Evaluation Analysis
This repository contains the scripts with following fatures:-
- Evaluating the performance of 2D Object Detection models. 
- Plotting the Precision and Recall curves.
- Comparing the AP values of all objects classes.
- OpenCV visualizer for visualizing the detections and ground truths of dataset.

## TODO
- [ ] Add Preview section.
- [x] Add requirement.txt file. 
- [x] Support for YOLOv9
- [x] Support for YOLOv11
- [ ] Support for mmdetection

## Getting started
1. Setup the environment
```
# Clone the repo
git clone --recursive https://github.com/gaurav00700/EvaluationAnalysis-2DObjectDetection.git

# Create the Python environment
conda create -n eval_img python=3.10 -y
conda activate eval_img

# Install requirements as per the framework (below is example of YOLOv9)
pip install -r ./submodules/YOLOv9/requirements.txt

```

2. Run visualization script (example) [YOLOv9](scripts/Img_EvaluateAnalysis.py)
```
python scripts/Img_EvaluateAnalysis.py --checkpoint <add checkpoint path> --model_name <model_name> --image_dir <image dataset dir> --ann_dir <annotation .jsons dataset dir>
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
│ ├── ...
├── scripts
│ ├── Img_EvaluateAnalysis.py
│ ├── ...
├── README.md
```
