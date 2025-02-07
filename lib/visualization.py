import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(parent_dir)  # add repo entrypoint to python path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import glob
import argparse
from typing import Union, Literal
from lib import utils_eval, tools

def viz_PrecisionRecall(data_dict: dict, class_decodings: dict, save_plot: bool = True, save_path: str = None):
    """Precision and recall visualization

    Args:
        data_dict (dict): dictionary containing precision and recall values of all classes
        class_decodings (dict): Decoding of object class from numbers to alphabets
        save_plot (bool, optional): Flag for saving the plot. Defaults to True.
        save_path (str, optional): Path to save the plot. Defaults to "./ap_plot.png".
    """
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left subplot: Precision and recall for all classes
    for cls, val in data_dict.items():
        if val['ap'] != 0.0:
            ax1.plot(val['recall'], 
                    val['precision'], 
                    label=f"{class_decodings[cls]}")
    ax1.set_xlim(0.0, 1.01)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True)
    ax1.legend(fontsize=7, loc='lower right')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title("Precision and Recall Plot")

    # Right subplot: Bars of AP of each class and mAP value as the last bar
    ap_values = [val['ap'] for val in data_dict.values() if val['ap'] !=0]
    class_labels = [class_decodings[cls] for cls in data_dict.keys() if data_dict[cls]['ap'] !=0]

    # Calculate mAP
    mAP = np.mean(ap_values)
    
    # Add mAP value as the last bar
    ap_values.append(mAP)
    class_labels.append('mAP')

    # ax2.bar(class_labels, ap_values)
    bars = ax2.bar(class_labels, ap_values)
    bars[-1].set_color('orange') # Color the mAP bar in orange
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel('Obj Class')
    ax2.set_ylabel('AP')
    ax2.set_title("AP of Each Class and mAP")

    # Show value of each bar on the top of the bar
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')
    
    # Add a dotted line for mAP
    # ax2.axhline(y=mAP, color='orange', linestyle='--')
    # ax2.text(len(class_labels) - 1, mAP + 0.01, f'mAP: {round(mAP, 2)}', color='orange', ha='center', va='bottom')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.suptitle("AP Plot")
    plt.tight_layout()

    # Save the plot if save_plot is True
    if save_plot:
        save_path= "ap_plot.png" if save_path is None else os.path.join(save_path,"ap_plot.png")
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to: {save_path}")

    plt.show()

def viz_predictions(
        gt_data:dict,
        pred_data:dict,
        imgs_path:list,
        save_path:str, 
        conf_thres:float= 0.0,
        **kwargs
        ) -> None:
        """Visualize predictions bbox with image using OpenCV

        Args:
            gt_data (dict): Dictionary containing ground truth data
            pred_data (dict): Dictionary containing predictions data
            imgs_path (list): List containing the list of image paths
            save_path (str, optional): path for saving the figure". Defaults to None.
            conf_thres (float, optional): Threshold for filtering predictions based on network confidence. Defaults to 0.0.
        """

        print(
                "=====================Keyboard Shortcuts==================",
                "A-> Previous Image",
                "D-> Next Image",
                "S-> Save image",
                "P-> Toggle Predictions",
                "G-> Toggle Ground Truth",
                "Q/Esc-> Quite", 
                "H-> Help",
                "============================End==========================",
                sep="\n"
            )
        
        # Params
        cL_offset = kwargs.get('cL_offset',50)
        color_gt= [255, 0, 0]
        color_pred= [0, 0, 255]
        current_idx = 0  # Start at the first image
        show_gt=True
        show_pred=True
        while True:
            img_path = imgs_path[current_idx]
            img_name = img_path.split(os.sep)[-1].split('.png')[0]

            # Read image
            img=cv2.imread(img_path)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W, C = img.shape # 540 x 960

            # Draw bounding boxes and display the class and confidence for Pred
            if show_pred and img_name in pred_data:

                # Frame data predictions
                preds = pred_data[img_name]

                # Skip if there is not prediction
                if len(preds) > 0:
                    for pred in preds:  

                        if pred['score'] < conf_thres:
                            continue

                        x1, y1, x2, y2 = map(lambda x:int(x), pred['pos_xyxy'])  # BBox corner points
                        # img = pred['image'] # Image                         
                        
                        # Draw the bounding box
                        cv2.rectangle(
                            img=img, 
                            pt1=(x1, y1), 
                            pt2=(x2, y2), 
                            color=color_pred, 
                            thickness=1
                        )

                        # Draw the cross inside the bounding box
                        if ('WBA' in pred) and (not pred['WBA']):
                            cv2.line(img, (x1, y1), (x2, y2), color_pred, 1)  # top-left to bottom-right
                            cv2.line(img, (x1, y2), (x2, y1), color_pred, 1)  # bottom-left to top-right
                        
                        # Add label on the image
                        cv2.putText(
                            img=img, 
                            text=f"Cls: {pred['cls_name']}, Conf: {pred['score']:.2f}, Area: {int(pred['bbox_area'])}", 
                            org=(x1, y1 - 10), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, 
                            color=color_pred, 
                            thickness=1
                        )
            
                else:                    
                    print(f"[WARN] No Prediction data for image '{img_path.split(os.sep)[-1]}'")
                
            # Draw bounding boxes and display the class for GT
            if show_gt and img_name in gt_data:
                
                # Frame data ground truth
                gts = gt_data[img_name] 

                # Skip if there is not prediction
                if len(gts) > 0:
                    for gt in gts:  
                        x1, y1, x2, y2 = map(lambda x:int(x), gt['pos_xyxy'])  # BBox corner points
                        
                        # Draw the bounding box
                        cv2.rectangle(
                            img=img, 
                            pt1=(x1, y1), 
                            pt2=(x2, y2), 
                            color=color_gt, 
                            thickness=1
                        )

                        # Draw the cross inside the bounding box
                        if ('WBA' in gt) and (not gt['WBA']):
                            cv2.line(img, (x1, y1), (x2, y2), color_gt, 1)  # top-left to bottom-right
                            cv2.line(img, (x1, y2), (x2, y1), color_gt, 1)  # bottom-left to top-right
                        
                        # Add label on the image
                        cv2.putText(
                            img=img, 
                            text=f"Cls: {gt['cls_name']}, Area: {int(gt['bbox_area'])}", 
                            org=(x1, y1 - 10), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, 
                            color=color_gt, 
                            thickness=1
                        )
                else:
                    print(f"[WARN] No Ground truth data for image '{img_path.split(os.sep)[-1]}'")
            
            # Draw centerline of Cartesian plane
            cv2.line(img, (W//2, 0), (W//2, H), [0,0,0], 1, cv2.LINE_AA) # Y-Y'
            # cv2.line(img, (0, H//2), (W, H//2), [0,0,0], 1, cv2.LINE_AA) # X-X'
            cv2.line(img, (W//2-cL_offset, 0), (W//2-cL_offset, H), [0,0,0], 1) # Negative offset Y-Y'
            cv2.line(img, (W//2+cL_offset, 0), (W//2+cL_offset, H), [0,0,0], 1) # Positive offset Y-Y'

            # Add labels for Prediction and Ground Truth
            cv2.putText(img, 'Prediction', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred, 2)
            cv2.putText(img, 'Ground Truth', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gt, 2)
            
            # Show the image
            cv2.imshow('Detection', img)
            
            # Wait for the user to press a key
            key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            
            # print(f"Key pressed is: {key}") # check the key press
            
            if key in [113, 27]:  # Q or ESC key to exit
                print("Exiting...")
                break

            elif key == 103: # G key to show GT
                show_gt = not show_gt
                print(f"GT visualization: {show_gt}")

            elif key == 112: # P key to show Pred
                show_pred = not show_pred
                print(f"Prediction visualization: {show_pred}")

            elif key == 97:  # A key --> previous image (if not first image)
                if current_idx > 0:
                    current_idx -= 1  # Decrease index
                else:
                    print("You are at the first image.")

            elif key == 100:  # D key --> next image (default behavior)
                if current_idx < len(imgs_path) - 1:
                    current_idx += 1  # Increase index
                else:
                    print("You are at the last image.")

            elif key == 115: # S key to save the image
                save_path_ = os.path.join(save_path, img_path.split(os.sep)[-1])
                cv2.imwrite(save_path_, img)
                print(f"Saved image to: {save_path_}")
            elif key == 104: # H key for help
                print(
                        "=====================Keyboard Shortcuts==================",
                        "A-> Previous Image",
                        "D-> Next Image",
                        "S-> Save image",
                        "P-> Toggle Predictions",
                        "G-> Toggle Ground Truth",
                        "Q/Esc-> Quite", 
                        "H-> Help",
                        "============================End==========================",
                        sep="\n"
                    )

        # Close all OpenCV windows
        cv2.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_dir', type=str, help='ground truth annotation .json file path')
    parser.add_argument('--pred_dir', type=str, help='prediction .json file path')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='NMS threshold value')
    parser.add_argument('--score_thres', type=float, default=0.0, help='Prediction Score threshold')
    parser.add_argument('--model_name', type=str, choices=["yolo2d_v11","mmdet2d"], default="yolo2d_v11", help='Model name to use')

    args = parser.parse_args()

    args.ann_dir="C:\\Users\\VL6CD7D\\Volkswagen AG\\CG Data Analytics T3-HF - General\\Austausch\\Internship-Gaurav\\GDC_ImageDetection\\data\\GDC images\\20241114-20241120_v1"
    args.pred_dir="data\\out\\1734075259\\prediction_results.json"

    # Annotation encoder and decoders
    class_encoder, class_decoder = utils_eval.class_encoder_and_decoder(args.model_name)

    # Load GT
    gt_data = utils_eval.parse_gt_jsons(args.ann_dir, class_encoder, class_decoder)

    # Load predictions
    with open(args.pred_dir) as f:
        pred_data = json.load(f)

    # Read images
    img_files=sorted(
            glob.glob(os.path.join(args.ann_dir,"*.png")),
            # key=lambda x: x.split(os.sep)[-1]  # Sort the images as per their name
            )

    viz_predictions(
        gt_data= gt_data,
        pred_data= pred_data,
        imgs_path= img_files,
        save_path= os.path.dirname(args.pred_dir),
        conf_thres= 0.0,
    )

def plot_object_count(obj_count_gt:dict, obj_count_pred:dict, save_plot: bool=True, save_path:Union[None, str]= None):
    """Plot showing the object classes counts between Ground Truth and Predictions

    Args:
        obj_count_gt (dict): Dictionary containing object counts for ground truth
        obj_count_pred (dict): Dictionary containing object counts for predictions
        save_plot (bool, optional): Flag to save the plot. Defaults to True.
        save_path (Union[None, str], optional): Path for saving the plot. Defaults to None.
    """
    # Bar positions
    labels = list(obj_count_gt.keys())
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    bars_gt = ax.bar(x - width/2, obj_count_gt.values(), width, label='GT')
    bars_pred = ax.bar(x + width/2, [obj_count_pred[k] for k in labels], width, label='Pred')

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Object class')
    ax.set_ylabel('Object count')
    ax.set_title('Object class count')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add labels on top of the bars
    for bar in bars_gt:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')
    for bar in bars_pred:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

    plt.xticks(rotation=45)
    fig.tight_layout()

    # Save plots
    if save_plot:
        save_path= "Object_cls_plot.png"if save_path is None else save_path
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to: {save_path}")

    plt.show()
