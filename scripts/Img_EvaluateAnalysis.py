import os, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from lib import evaluation_img, utils_eval

def parse_args():
    """Method for creating argument parser"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="yolo11x.pt", help="path of Checkpoint file")
    parser.add_argument("--device", default="cpu", help="Device used for inferences")
    parser.add_argument("--model_name",type=str, choices=["yolo2d", "mmdet2d"], required=True, help="Name of detector")
    parser.add_argument("--save_dir",type=str, help="Directory for saving result")
    parser.add_argument("--image_dir",type=str, required=True, help="Directory for dataset images")
    parser.add_argument("--ann_dir", type=str, required=True, help="Directory for 2D annotations .jsons")
    args = parser.parse_args()
    
    return args

# Argument parser
args = parse_args()

# Detection frameworks
detectors = {
    "yolo2d":evaluation_img.YOLO_detection,
    "mmdet2d":evaluation_img.MMdet_detection,
}

# Initialize tester class
tester = detectors[args.model_name](args)

# Run prediction
tester.prediction(save_result=True, vis_pred=True)
tester.evaluate(save_result=True, vis_AP=True)