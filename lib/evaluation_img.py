import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(parent_dir)  # add repo entrypoint to python path
import numpy as np
import glob
from collections import defaultdict
import tqdm
from PIL import Image 
import json
import time
import torch
from pathlib import Path
from typing import Literal, Union
from lib import utils_eval, visualization, tools

# YOLO related Imports 
try:
    sys.path.append(os.path.join(parent_dir, 'submodules', 'YOLOv9'))  # add repo entrypoint to python path
    import hydra
    from lightning import Trainer
    from yolo.config.config import Config
    from yolo.tools.solver import InferenceModel
    from yolo.utils.logging_utils import setup
    from yolo import (
        AugmentationComposer,
        PostProcess,
        create_converter,
        create_model,
        draw_bboxes,
    )
except:
    print("Failed to import YOLO v9 packages")

def model_class(model:Literal["yolo2d_v11", "yolo2d_v9", "mmdet2d"]):
    """Select the model class based on model name"""
    
    # Detection frameworks
    detectors = {
        "yolo2d_v11":YOLOv11_detection,
        "yolo2d_v9":YOLOv9_detection,
        "mmdet2d":MMdet_detection,
    }

    return detectors[model]

class Img_TestEval:
    def __init__(self, args, **kwargs) -> None:

        self.args = args
        self.model = None # initialize the model
        self.device = args.device
        
        # expand kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not getattr(self, 'bbox_area_thres', None):
            print("[WARN] 'bbox_area_thres' not found, falling back to default values: {}")
            self.bbox_area_thres = {}
        if not getattr(self, 'img_HxW', None):
            print(f"[WARN] 'img_HxW' not found, falling back to default values: {(540, 960)}")
            self.img_HxW=(540, 960)

        # For storing results
        self.predict_results = defaultdict(dict)
        self.gt_data = defaultdict(dict)
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

        # Read image files and Gt annotation files
        self.read_data()
        
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
        
        # Annotation encoder and decoders
        class_encoder, class_decoder = utils_eval.class_encoder_and_decoder(self.args.model_name)

        # Load GT annotations
        self.gt_data = utils_eval.parse_gt_jsons(self.args.ann_dir, class_encoder, class_decoder)

        # Enrich GT with WBA flag
        self.gt_data = utils_eval.WBA_check(
            data=self.gt_data, 
            bbox_area_thres=self.bbox_area_thres,
            img_HxW=self.img_HxW
            )
        
        # reset results
        self.reset_test_val()  # reset results
    
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
            json.dump(self.predict_results, f, indent=4, cls=tools.NpEncoder)

        print(f"Prediction results saved at: {file_path}")

    def evaluate(
        self,
        nms_thres:float=0.5,
        score_thres:float=0.0,
        WBA_filter:bool=False,
        save_result:bool=True,
        vis_AP:bool=True,
    ) -> None:
        """Evaluate the predictions using ground truth

        Args:
            nms_thres (float, optional): NMS threshold for iou. Defaults to 0.5.
            score_thres (float, optional): Threshold for prediction score. Defaults to 0.0.
            WBA_filter (bool, optional): Flag for filtering Bboxes as per WBA criteria. Defaults to False.
            save_result (bool, optional): Save validation result. Defaults to True.
            viz_AP (bool, optional): Visualize AP plot. Defaults to True.
        """
        assert self.test_flag, "Run prediction() before validating"

        if self.args.WBA_filter:
            WBA_filter = True

        self.AP_sequence, self.AP_data_seq = utils_eval.validate(
            gt= utils_eval.rm_non_WBA(self.gt_data) if WBA_filter else self.gt_data,  # ground truth annotations json
            pred_data=utils_eval.rm_non_WBA(self.predict_results) if WBA_filter else self.predict_results,  # predictions
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
            json.dump(val_results, f, indent=4 , cls=tools.NpEncoder)

        print(f"Validation results saved at: {file_path}")

    def viz_pres_recall(self, save_plot:bool=True, save_dir:str="") -> None:
        """Visualize precision recall curve"""

        # Annotation encoder and decoders
        class_encoder, class_decoder = utils_eval.class_encoder_and_decoder(self.args.model_name)

        if len(self.AP_sequence) > 0:
            visualization.viz_PrecisionRecall(self.AP_sequence, class_decoder, save_plot, self.save_dir)
        
        else:
            print("No predictions available. Run prediction() first.")

    def viz_predictions(self, save_path:Union[str, None]= None, conf_thres:float= 0.0) -> None:
        """Visualize predictions bbox with point cloud

        Args:
            save_path (str, optional): path for saving the figure". Defaults to None.
            conf_thres (float, optional): Threshold for filtering predictions based on network confidence. Defaults to 0.0.
        """
        save_path= self.save_dir if save_path is None else save_path

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

class YOLOv11_detection(Img_TestEval):
    """YOLO Object detection"""

    def __init__(self, args, **kwargs):
        super().__init__(args,**kwargs)  # Initialize Img_TestEval class
        
        # YOLO related Imports
        try: 
            from ultralytics import YOLO
        except ValueError: 
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
                ann = {
                        "score": score.tolist(), 
                        "pos": pos.tolist(), 
                        "pos_xyxy":pos_xyxy.tolist(),
                        "bbox_area": tools.bbox2d_area(pos.tolist()),
                        "cls": int(cls),
                        "cls_name":results.names[int(cls)],
                    }
                
                # Enrich prediction with WBA flag
                ann = utils_eval.WBA_check(
                        data={'temp':[ann]}, 
                        bbox_area_thres=self.bbox_area_thres,
                        img_HxW=self.img_HxW
                        )['temp'][0]
                
                detection.append(ann)

            # Add predictions to dict
            self.predict_results[str(frame_name)] = detection

class YOLOv9_detection(Img_TestEval):
    """YOLOv9 Object detection"""

    def __init__(self, args, **kwargs):
        super().__init__(args,**kwargs)  # Initialize Img_TestEval class

        # Load model configurations using hydra
        hydra.initialize(config_path="../data/in/yolov9/config", version_base= None)
        self.cfg: Config = hydra.compose(config_name="config", overrides=[]) # change the model name in the config file

        # Overrides configs
        self.cfg.weight = args.checkpoint
        self.cfg.task.data.source = args.image_dir
        self.cfg.out_path = self.save_dir
        
        # initialize the model and trainer
        self.trainer, self.model = self.load_model(self.cfg) 
        
        # Inference
        # self.trainer.predict(self.model)

    def load_model(self, cfg: dict):        

        callbacks, loggers, save_path = setup(cfg)

        trainer = Trainer(
            accelerator="auto",
            max_epochs=getattr(cfg.task, "epoch", None),
            precision='bf16-mixed', #"16-mixed",
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=1,
            gradient_clip_val=10,
            gradient_clip_algorithm="value",
            deterministic=True,
            enable_progress_bar=not getattr(cfg, "quite", False),
            default_root_dir=save_path,
        )

        model = InferenceModel(cfg)
        # trainer.predict(model)
        return trainer, model

    def inference(self):

        # Setup the helpers
        # self.trainer.predict(self.model)
        transform = AugmentationComposer([], self.cfg.image_size)
        converter = create_converter(self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device)
        post_proccess = PostProcess(converter, self.cfg.task.nms)

        # draw_bboxes(pil_image, pred_bbox, idx2label=self.cfg.dataset.class_list)

        for img_file in tqdm.tqdm(self.img_files, desc="Predicting Images"):
            
            frame_name = os.path.basename(img_file).split(".")[0] # File name
            pil_image = Image.open(img_file) # Load image
            image, bbox, rev_tensor = transform(pil_image) # Transform the image
            image = image.to(self.device)[None] # Move to device
            rev_tensor = rev_tensor.to(self.device)[None]
            
            # Model inference
            with torch.no_grad():
                predict = self.model(image)
                results = post_proccess(predict, rev_tensor) # pred_bbox

                '''results (List of Lists/Tensors): Bounding boxes with [class_id, x_min, y_min, x_max, y_max],
                    where coordinates are normalized [0, 1]'''
            # draw_bboxes(pil_image, results, idx2label=cfg.dataset.class_list)
            # if len(results) == 0:
            #     continue

            detection = list()
            # if isinstance(results, list) or results.ndim == 3:
            if results[0].shape[0] > 0:

                # List of predictions
                for bbox in results[0]:
                    class_id, x_min, y_min, x_max, y_max, conf = map(float, bbox)
                    pos_xyxy = [x_min, y_min, x_max, y_max]
                    pos_xywh = tools.xyxy_to_xywh([x_min, y_min, x_max, y_max])

                    ann = {
                            "score": conf, 
                            "pos": pos_xywh, 
                            "pos_xyxy":pos_xyxy,
                            "bbox_area": tools.bbox2d_area(pos_xywh),
                            "cls": int(class_id),
                            "cls_name":self.cfg.dataset.class_list[int(class_id)],
                        }
                    
                    # Enrich prediction with WBA flag
                    ann = utils_eval.WBA_check(
                            data={'temp':[ann]}, 
                            bbox_area_thres=self.bbox_area_thres,
                            img_HxW=self.img_HxW
                            )['temp'][0]
                    
                    detection.append(ann)

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
                ann = {
                        "score": score.tolist(), 
                        "pos": pos.tolist(), 
                        "pos_xyxy":pos_xyxy.tolist(),
                        "bbox_area": tools.bbox2d_area(pos.tolist()),
                        "cls": int(cls),
                        "cls_name":results.names[int(cls)],
                    }
                detection.append(ann)

            # Add predictions to dict
            self.predict_results[str(frame_name)] = detection
