import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(parent_dir)  # add repo entrypoint to python path
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import argparse
from collections import defaultdict
import tqdm
import json
import time
from typing import Literal, Union
from lib import utils_eval, visualization

class Img_TestEval:
    def __init__(self, args) -> None:

        self.args = args
        self.model = None # initialize the model

        # For storing results
        self.predict_results = defaultdict(dict)
        self.AP_sequence = dict()
        self.AP_data_seq = dict()

        # Directory for saving data
        if not args.save_dir:
            self.save_dir=os.path.join('data', 'out', str(int(time.time())))
        else:
            self.save_dir=args.save_dir

        # Flags
        self.test_flag = False
        self.val_flag = False

        # Read files
        self.img_files=self.read_data()
        
        # Actions:
        # Sanity check
        if len(self.img_files) > 0:
            print(f"[INFO]: {len(self.img_files)} Images found")
        else:
            assert len(self.img_files) > 0, "No images found"
        
        # Directory for saving data
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

    def read_data(self) -> list:
        """Read the images"""
        
        # read image files
        self.img_files=sorted(
            glob.glob(os.path.join(self.args.image_dir,"*.png")),
            # key=lambda x: x.split(os.sep)[-1]  # Sort the images as per their name
            )

        # reset results
        self.reset_test_val()  # reset results

        return self.img_files
    
    def inference(self):
        pass

    def prediction(self, save_result:bool= True, vis_pred:bool= True) -> None:
        """Run prediction on all pcd files and store results"""
        
        assert not self.test_flag, "Already in test mode, Run reset_test_val() to reset"

        # Infer the model
        self.inference()

        self.test_flag = True

        # Save Predictions
        if save_result:
            self.save_prediction()
        
        # Visualize predictions
        if vis_pred:
            self.viz_predictions(
                save_path=self.save_dir,
                # conf_thres= 0.0
            )        

    def save_prediction(self, file_path:str=None) -> None:
        """Save prediction result in json file

        Args:
            file_path (str, optional): file path. Defaults to None.
        """
        
        assert self.test_flag, "Run prediction() before saving predictions"
        
        # save prediction results in json file
        if file_path is None:
            file_path = os.path.join(self.save_dir,"prediction_results.json")
        with open(file_path, "w") as f:
            json.dump(self.predict_results, f, indent=4, cls=utils_eval.NpEncoder)

        print(f"Prediction results saved at: {file_path}")

    def evaluate(
        self,
        nms_thres:float=0.5,
        score_thres:float=0.0,
        save_result:bool=True,
        vis_AP:bool=True,
    ) -> None:
        """Evaluate the predictions using ground truth

        Args:
            nms_thres (float, optional): NMS threshold for iou. Defaults to 0.5.
            score_thres (float, optional): Threshold for prediction score. Defaults to 0.0.
            save_result (bool, optional): Save validation result. Defaults to True.
            viz_AP (bool, optional): Visualize AP plot. Defaults to True.
        """
        assert self.test_flag, "Run prediction() before validating"

        self.AP_sequence, self.AP_data_seq = utils_eval.validate(
            gt_dir=self.args.ann_dir,  # ground truth annotations json
            pred_data=self.predict_results,  # predictions
            nms_method='iou',
            nms_thres=nms_thres,  # non-max suppression threshold for iou
            score_thres=score_thres,
            model_name=self.args.model_name,
        )
        
        self.val_flag = True

        # Save validation result in json
        if save_result:
            self.save_validation()
        
        # Visualize AP plot
        if vis_AP:
            self.viz_pres_recall()
        

    def save_validation(self, file_path:str= None) -> None:
        """Save validation results to json file

        Args:
            file_path (str, optional): path for saving json file. Defaults to None.
        """

        assert (
            self.val_flag and len(self.AP_sequence.keys()) > 0
        ), "Run validate() before saving validation results"

        # file path
        if file_path is None:
            file_path = os.path.join(self.save_dir,"validation_results.json")
        
        # Reformat data for json serialization
        ap_seq = {str(k): v for k, v in self.AP_sequence.items()} # convert cls keys to strings
        ap_data = defaultdict(dict) # convert cls keys to string 
        for f in self.AP_data_seq:
            for k, v in self.AP_data_seq[f].items():
                ap_data[f][str(k)] = v
        
        val_results = {
            'AP_seq': ap_seq,
            'AP_data_seq': ap_data 
            }
        
        # Save data to json file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(val_results, f, indent=4 , cls=utils_eval.NpEncoder)

        print(f"Validation results saved at: {file_path}")

    def viz_pres_recall(self, save_plot:bool=True, save_dir:str="") -> None:
        """Visualize precision recall curve"""

        # Annotation encoder and decoders
        class_encoder, class_decoder = utils_eval.class_encoder_and_decoder(self.args.model_name)

        if len(self.AP_sequence.keys()) > 0:
            visualization.viz_PrecisionRecall(self.AP_sequence, class_decoder, save_plot, self.save_dir)
        
        else:
            print("No predictions available. Run prediction() first.")

    def viz_predictions(self, save_path:Union[str, None]= None, conf_thres:float= 0.0) -> None:
        """Visualize predictions bbox with point cloud

        Args:
            save_path (str, optional): path for saving the figure". Defaults to None.
            conf_thres (float, optional): Threshold for filtering predictions based on network confidence. Defaults to 0.0.
        """

        # Annotation encoder and decoders
        class_encoder, class_decoder = utils_eval.class_encoder_and_decoder(self.args.model_name)

        # Load GT
        self.gt_data = utils_eval.parse_gt_jsons(self.args.ann_dir, class_encoder, class_decoder)

        visualization.viz_predictions(
            gt_data= self.gt_data,
            pred_data= self.predict_results,
            imgs_path= self.img_files,
            save_path= save_path,
            conf_thres= conf_thres,
        )

    def reset_val(self) -> None:
        """Reset validation results"""

        self.AP_sequence = {}
        self.AP_data_seq = {}
        self.val_flag = False
        print("==>Reset validation results")

    def reset_test_val(self) -> None:
        """Reset prediction, validation and visualization data"""

        # self.img_files = []
        self.predict_results = defaultdict(dict)
        self.AP_sequence = {}
        self.AP_data_seq = {}
        self.test_flag = False
        self.val_flag = False
        print("==>Reset prediction, validation and visualization data")


class YOLO_detection(Img_TestEval):
    """YOLO Object detection"""

    def __init__(self, args):
        super().__init__(args)  # Initialize Img_TestEval class
        
        # YOLO related Imports
        try: 
            from ultralytics import YOLO
        except: 
            print("YOLO not found")

        self.model = YOLO(args.checkpoint) # initialize the model

    def inference(self):

        for img_file in tqdm.tqdm(self.img_files, desc="Predicting Images"):
            
            # Model inference
            results = self.model(img_file)[0]

            # List of predictions
            scores_pred = results.boxes.conf.detach().cpu().numpy()
            bboxes_2d_pred = results.boxes.xywh.detach().cpu().numpy() # xywh & xyxy are available
            bboxes_2d_xyxy_pred = results.boxes.xyxy.detach().cpu().numpy()
            labels_2d_pred = results.boxes.cls.detach().cpu().numpy() # encoded class
            # labels_2d_pred = results.names[labels_2d_pred] # decoded class
        
            # File name
            frame_name = os.path.basename(results.path).split(".")[0]

            # Append result to the list
            detection = list()
            for score, pos, pos_xyxy, cls in zip(
                scores_pred, 
                bboxes_2d_pred, 
                bboxes_2d_xyxy_pred, 
                labels_2d_pred
                ):
                detection.append(
                    {
                        "score": score.tolist(), 
                        "pos": pos.tolist(), 
                        "pos_xyxy":pos_xyxy.tolist(),
                        "cls": int(cls),
                        "cls_name":results.names[int(cls)],
                    }
                )

            # Add predictions to dict
            self.predict_results[str(frame_name)] = detection

class MMdet_detection(Img_TestEval):
    """MMdetection Object detection"""

    def __init__(self, args):
        super().__init__(args)  # Initialize Img_TestEval class

        # MMdetection related Imports
        try: 
            from mmdet.apis import DetInferencer
        except: 
            print("MMdet_detection not found")

        # Initialize the DetInferencer
        self.model = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', weights='path/to/rtmdet.pth')

        # Perform inference
        # inferencer('demo/demo.jpg', show=True)

    def inference(self):

        for img_file in tqdm.tqdm(self.img_files, desc="Predicting Images"):
            
            # Model inference
            results = self.model(img_file)[0]

            # List of predictions
            scores_pred = results.boxes.conf.detach().cpu().numpy()
            bboxes_2d_pred = results.boxes.xywh.detach().cpu().numpy() # xywh & xyxy are available
            bboxes_2d_xyxy_pred = results.boxes.xyxy.detach().cpu().numpy()
            labels_2d_pred = results.boxes.cls.detach().cpu().numpy() # encoded class
            # labels_2d_pred = results.names[labels_2d_pred] # decoded class
        
            # File name
            frame_name = os.path.basename(results.path).split(".")[0]

            # Append result to the list
            detection = list()
            for score, pos, pos_xyxy, cls in zip(
                scores_pred, 
                bboxes_2d_pred, 
                bboxes_2d_xyxy_pred, 
                labels_2d_pred
                ):
                detection.append(
                    {
                        "score": score.tolist(), 
                        "pos": pos.tolist(), 
                        "pos_xyxy":pos_xyxy.tolist(),
                        "cls": int(cls),
                        "cls_name":results.names[int(cls)],
                    }
                )

            # Add predictions to dict
            self.predict_results[str(frame_name)] = detection