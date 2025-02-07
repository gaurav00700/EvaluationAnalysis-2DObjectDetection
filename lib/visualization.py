import os
import time
from datetime import datetime
import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

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
        conf_thres:float= 0.0
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
                        
                        # Add label on the image
                        cv2.putText(
                            img=img, 
                            text=f"Class: {pred['cls_name']}, Conf: {pred['score']:.2f}", 
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
                        
                        # Add label on the image
                        cv2.putText(
                            img=img, 
                            text=f"Class: {gt['cls_name']}", 
                            org=(x1, y1 - 10), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, 
                            color=color_gt, 
                            thickness=1
                        )
                else:
                    print(f"[WARN] No Ground truth data for image '{img_path.split(os.sep)[-1]}'")

            
            # Add labels for Prediction and Ground Truth
            cv2.putText(img, 'Prediction', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred, 1)
            cv2.putText(img, 'Ground Truth', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gt, 1)
            
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
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_dir', type=str, help='ground truth annotation .json file path')
    parser.add_argument('--pred_dir', type=str, help='prediction .json file path')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='NMS threshold value')
    parser.add_argument('--score_thres', type=float, default=0.0, help='Prediction Score threshold')
    parser.add_argument('--model_name', type=str, choices=["yolo2d","mmdet2d"], default="yolo2d", help='Model name to use')

    args = parser.parse_args()

    args.ann_dir= "Add dir here"
    args.pred_dir= "Add dir here"

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
