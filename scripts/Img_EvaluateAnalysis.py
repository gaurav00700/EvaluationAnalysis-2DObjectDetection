import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../.."))) # add path to sys path
import argparse
from lib import evaluation_img, utils_eval

def parse_args(args_list:list=None):
    """Method for creating argument parser"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path of Checkpoint file (.pt)")
    parser.add_argument("--device", default="cpu", help="Device used for inferences")
    parser.add_argument("--model_name",type=str, choices=["yolo2d_v11","yolo2d_v9", "mmdet2d"], required=True, help="Name of object detection framework")
    parser.add_argument("--save_dir",type=str, help="Directory for saving result")
    parser.add_argument("--image_dir",type=str, required=True, help="Directory for dataset images")
    parser.add_argument("--ann_dir", type=str, required=True, help="Directory for 2D annotations .jsons")
    
    return parser.parse_args(args_list) if args_list else parser.parse_args() 

# Argument parser
args = parse_args()

# initialize the model class
detector = evaluation_img.model_class(args.model_name)(args)

# Run prediction and evaluations
detector.prediction(save_result=True, vis_pred=True)
detector.evaluate(save_result=True, vis_AP=True)

print("DONE !!!")