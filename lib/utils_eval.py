import numpy as np
import os
import glob
import json
import copy
import matplotlib; matplotlib.use('Qt5Agg')  # Set the backend to 'Qt5Agg'
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Literal, Union

class NpEncoder(json.JSONEncoder):
    """convert nested numpy to nested list

    Args:
        json (json class): Extensible JSON <http://json.org> encoder for Python data structures.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return super(NpEncoder, self).default(obj)

def xywh_to_xyxy(box):
    """
    Convert xywh format (center x, center y, width, height) to (x_min, y_min, x_max, y_max).

    Args:
        box (list or array): Bounding box in YOLO format [center_x, center_y, width, height]

    Returns:
        list: Bounding box in (x_min, y_min, x_max, y_max) format
    """
    center_x, center_y, width, height = box
    x_min = center_x - (width / 2)
    y_min = center_y - (height / 2)
    x_max = center_x + (width / 2)
    y_max = center_y + (height / 2)
    
    return [x_min, y_min, x_max, y_max]

def xyxy_to_xywh(box):
    """
    Convert (x_min, y_min, x_max, y_max) format to xywh format (center x, center y, width, height).

    Args:
        box (list or array): Bounding box in (x_min, y_min, x_max, y_max) format

    Returns:
        list: Bounding box in YOLO format [center_x, center_y, width, height]
    """
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [center_x, center_y, width, height]

def parse_gt_jsons(dir:str, cls_encoder:dict, cls_decoder:dict) -> dict:
    """Parse annotations files

    Args:
        dir (str): Path of directory
        cls_encoder (dict): Annotation object classes encoder
        cls_decoder (dict): Annotation object classes decoder

    Returns:
        dict: Dictionary containing parse json files
    """
    
    # Read image files
    ann_files=sorted(glob.glob(os.path.join(dir,"*.json")))

    # Load image annotations 
    gt_data = defaultdict(dict)
    for ann_file in ann_files:

        # Read jsons
        with open(ann_file) as f:
            anns_data = json.load(f)

        # for ann_data in anns_data['shapes']:
        object = list()
        for ann_data in anns_data['shapes']:
            pos_xyxy= sum(ann_data['points'], []) # (2,2) -> (1,4)                
            object.append(
                {
                    "pos": xyxy_to_xywh(pos_xyxy),
                    "pos_xyxy": pos_xyxy,
                    "cls": cls_encoder[ann_data['label']],
                    "cls_name": ann_data['label'],
                }
            )
        gt_data[anns_data['imagePath'].split(".")[0]] = object
    
    return gt_data

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    # Calculate the area of intersection rectangle
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    # Calculate the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def tp_mask_img(frame_value: dict, iou_thresh: float = 0.5) -> dict:
    """Calculates true positive for a given IoU threshold and prediction over a single frame for image data.

    Args:
        frame_value (dict): dict consists of predictions and annotations
        iou_thresh (float, optional): IoU threshold for associating ground truth with predictions. Defaults to 0.5.

    Returns:
        tp_dict (dict): true positive mask with object ids as keys and boolean mask as values
        idx (list): indices after sorting predictions by score
    """ 
    # Unpack the data
    ann_poses = frame_value['annotation']['position']  # Assuming 'bbox' contains [x1, y1, x2, y2]
    pred_poses = frame_value['prediction']['position']  # Assuming 'bbox' contains [x1, y1, x2, y2]
    pred_scores = frame_value['prediction']['tracking_score']
    pred_ids = frame_value['prediction']['tracking_id']
    
    # Sorting in descending order
    idx = np.argsort(pred_scores)[::-1]  # get indices for sorting
    pred_poses = pred_poses[idx]  # sort boxes
    idx = list(idx)
    
    # Initializing all detection with false positive
    tp_dict = {id: 0 for id in pred_ids}

    if ann_poses.shape[0]:
        for count, pred_pose in enumerate(pred_poses):
            ious = np.array([calculate_iou(pred_pose, ann_pose) for ann_pose in ann_poses])
            max_iou_idx = np.argmax(ious)  # index of the highest IoU
            
            if ious[max_iou_idx] >= iou_thresh:
                pred_id = frame_value['prediction']['tracking_id'][idx[count]]  # prediction id using index
                tp_dict[pred_id] = 1  # Object associated with a gt

                # Discard gt by setting it to a large value
                ann_poses[max_iou_idx, :] = np.array([1e8, 1e8, 1e8, 1e8]) 

    return tp_dict , idx

def cal_ap(score: np.array, gt_no: int, true_pos: np.array):
    """Calculate average precision score

    Args:
        score (np.array): Prediction score
        gt_no (int): number of ground truth values
        true_pos (np.array): True positive values

    Returns:
        _type_: Average precision, precision, recall
    """
    idx = np.argsort(score)[::-1] # indices of sorting in descending order
    true_pos = true_pos[idx] # sort in descending order

    # Calculate precision and recall
    Precision = np.cumsum(true_pos)/np.linspace(1, len(true_pos), len(true_pos))
    Recall = np.cumsum(true_pos)/ gt_no

    # Add a zero at the beginning of Recall and Precision for the first interval
    Recall = np.concatenate(([0], Recall))
    Precision = np.concatenate(([1], Precision))

    # Ensure no NaN values in Precision and Recall
    Precision = np.nan_to_num(Precision)
    Recall = np.nan_to_num(Recall)

    # Area under precision and recall curve
    ap = np.trapezoid(Precision, Recall) 

    return ap, Precision, Recall

def class_encoder_and_decoder(model_name:Literal['nuscenes, kitti', 'yolo2d']):
    """Encoder and decoder as per model type
    Args:
        model_name (Literal['nuscenes', 'kitti', 'yolo2d']): model_name name for pointpillars

    Returns:
        _type_: dict of encoder and decoder of kitti
    """
    #mmdetection classes
    #ref: mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py
    # mm3ddet_classes = ['Pedestrian', 'Cyclist', 'Car']
    mm3ddet_classes = ['Car']
    mm3ddet_classes = [cls.lower() for cls in mm3ddet_classes]

    # Encoding of Kitti360 classes to nuScenes classes
    kitti_classes = ['car', 'truck', 'trailer', 'bus', 'bicycle', 'motorcycle', 'person' , 'caravan',  'unknown vehicle' ]

    # Nuscene class encoding
    nus_classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']  

    # Yolo2d class encoding as per COCO dataset
    yolo2d_classes={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    #kitti class encoding for nuscenes
    if model_name == 'nuscenes':    
        # ancoded classes
        source_class_encodings = {class_: i for i, class_ in enumerate(nus_classes)}
        
        # decoded classes
        source_class_decodings = {encode: cls for cls, encode in source_class_encodings.items()}    

        # Mapping of custom obj classes to mm3ddet obj classes
        mapping = {
            'person': 'pedestrian',
        }

        # update encoding as per custom classes
        for from_, to_ in mapping.items():
            val_ = source_class_encodings[to_] if to_ in source_class_encodings else -1
            source_class_encodings.update({from_: val_})
    
    #kitti class encoding as per mmdetection kitti pointpillars configs
    elif model_name == 'kitti':
        # ancoded classes
        source_class_encodings = {class_: i for i, class_ in enumerate(mm3ddet_classes)}
        
        # decoded classes
        source_class_decodings = {encode: cls for cls, encode in source_class_encodings.items()} 

        # Mapping of custom obj classes to mm3ddet obj classes
        mapping = {
            'person': 'pedestrian',
            'bicycle': 'cyclist',
            'motorcycle': 'cyclist',
            'truck': 'car',
            'trailer': 'car',
            'bus': 'car',
            'caravan': 'car',
            'unknown vehicle': 'car',
        }

        # update encoding as per custom classes
        for from_, to_ in mapping.items():
            val_ = source_class_encodings[to_] if to_ in source_class_encodings else -1
            source_class_encodings.update({from_: val_})

    elif model_name=='yolo2d':
        # Encoded classes
        source_class_encodings = {class_: i for i, class_ in yolo2d_classes.items()}
            
        # Decoded classes
        source_class_decodings = copy.deepcopy(yolo2d_classes)
        
        # Mapping of custom obj classes to COCO obj classes
        mapping = {
            'van': 'car',
            'tractor': 'truck'
        }
        # mapping = {
        #     'van': 'car',
        #     # 'motorcycle': 'person',
        #     'bicyclist': 'bicycle',
        #     'motorcyclist': 'motorcycle',
        #     'tractor': 'truck'
        # }

        # update encoding as per custom classes
        for from_, to_ in mapping.items():
            val_ = source_class_encodings[to_] if to_ in source_class_encodings else -1
            source_class_encodings.update({from_: val_})
        
    else:
        raise ValueError(f"Invalid model_name: {model_name}")

    return source_class_encodings, source_class_decodings

def cal_AP_img_sequence(
        gt_data, 
        pred_data,
        nms_method:Literal['iou']='iou', 
        nms_thres=0.5, 
        score_thres=0.0,
        ) -> dict:
    """Calculate mean Average Precision for all frames (sequence) for images

    Args:
        - gt_data (dict): data dictionary in prescribed format
        - pred_data (dict): data dictionary dictionary in prescribed format
        - nms_method (str): method for NMS
        - nms_thres (float): distance threshold for NMS
        - score_thres (float): confidence threshold

    Returns:
        - ap_clss (dict): average precision of all classes in  sequence
        - tp_data_seq (dict): true positive values for each frame
    """
    tp_data_clss = dict()
    tp_data_seq = defaultdict(dict)
    for frame in pred_data:

        # Filter prediction
        
        pred_score = np.array([obj['score'] for obj in pred_data[frame]]) # score        
        pred_pos = np.array([obj['pos_xyxy'] for obj in pred_data[frame]]) # position        
        pred_cls = np.array([obj['cls'] for obj in pred_data[frame]]) # obj class

        # Indices meeting the criteria
        idx = np.where(pred_score >= score_thres)[0]

        # Final data
        pred_score = pred_score[idx]
        pred_pos = pred_pos[idx]
        pred_cls = pred_cls[idx]
        
        # Filter GT
        gt_pos = np.array([obj['pos_xyxy'] for obj in gt_data[frame]]) # position        
        gt_cls = np.array([obj['cls'] for obj in gt_data[frame]]) # filter data using score indices

        # Restructure ground truth and prediction data for TP mask calculation
        # frame_cls_value = dict()
        for cls in np.unique(pred_cls):
            
            # Index of current object class
            ann_idx = np.where(gt_cls == cls)[0] # ground truth
            pred_idx = np.where(pred_cls == cls)[0] # prediction

            # Format data for TP calculation
            frame_cls_value = {
                'annotation': {'position': gt_pos.reshape(-1,4)[ann_idx]}, 
                'prediction': {'position': pred_pos[pred_idx],
                               'tracking_score': pred_score[pred_idx], 
                               'tracking_id': pred_idx},
                }
        
            # Calculate TP mask for all classes of the frame
            tp_dict, sorted_idx = tp_mask_img(frame_cls_value, nms_thres)
            score = frame_cls_value['prediction']['tracking_score']
            gt_num = frame_cls_value['annotation']['position'].shape[0]

            # Append data for each frame
            if cls in tp_data_clss:
                tp_data_clss[cls]['tp'] = np.concatenate((tp_data_clss[cls]['tp'], list(tp_dict.values()) ))
                tp_data_clss[cls]['score'] = np.concatenate((tp_data_clss[cls]['score'], score ))
                tp_data_clss[cls]['gt_num'] += gt_num
            else:
                tp_data_clss.update({cls: {'tp': np.asarray(list(tp_dict.values())),
                                           'score': score,
                                           'gt_num': gt_num} })

            # Append tp data for the frame
            tp_data_seq[frame][cls] = {'id': list(tp_dict.keys()),
                                       'tp': list(tp_dict.values()),
                                       'score': score,
                                       'position': frame_cls_value['prediction']['position'],
                                       'gt_num': gt_num}

    # Calculate AP for all classes of all frames
    ap_clss = dict()
    for cls in tp_data_clss:
        # Calculate average precision
        ap, precision, recall = cal_ap(tp_data_clss[cls]['score'], 
                                       tp_data_clss[cls]['gt_num'], 
                                       tp_data_clss[cls]['tp'])
        # Append data for each class
        ap_clss.update({cls: {'ap': ap, 
                              'precision': precision, 
                              'recall': recall,
                              'gt_num': tp_data_clss[cls]['gt_num']} })

    return ap_clss, tp_data_seq

def validate(
        gt_dir:str, 
        pred_data:Union[dict,str], 
        nms_method:Literal['iou']='iou',
        nms_thres=0.5, 
        score_thres=0.0, 
        model_name:Literal['nuscenes','kitti','yolo2d']='yolo2d'
    ) -> dict:
    """Validation the predictions

    Args:
        - gt_dir (str): Directory containing annotation files 
        - pred_data (Union[dict,str]): Prediction data in dictionary or path of json file
        - nms_method (Literal[&#39;iou&#39;], optional): Method for NMS. Defaults to 'iou'.
        - nms_thres (float, optional): Threshold for NMS. Defaults to 0.5.
        - score_thres (float, optional): Score threshold for filtering predictions . Defaults to 0.0.
        - model_name (Literal[&#39;nuscenes&#39;,&#39;kitti&#39;,&#39;yolo2d&#39;], optional): Name of Object detection Model. Defaults to 'yolo2d'.

    Returns:
        - dict: AP for each class and TP mask for all images
    """

    # Annotation encoder and decoders
    class_encodings, class_decodings = class_encoder_and_decoder(model_name)

    # Parse GT
    if model_name == 'yolo2d':
        gt_data = parse_gt_jsons(gt_dir, class_encodings, class_decodings)
    else:     # Load from json file
        with open(gt_dir) as f:
            gt_data = json.load(f)

    # Load prediction data from json file
    if isinstance(pred_data, str):
        print(f'Loading prediction data from: {pred_data}')
        with open(pred_data) as f:
            pred_data = json.load(f)

    # AP of all classes in the sequence
    AP_sequence, tp_data_seq = cal_AP_img_sequence(
        gt_data=gt_data, 
        pred_data=pred_data, 
        nms_method='iou', 
        nms_thres=nms_thres,
        score_thres=score_thres, 
        )
    
    # Print AP mAP of all classes of sequence
    print("AP of classes of sequence:{ap}".format(
        ap= {class_decodings[int(cls)]: 
             round(AP_sequence[cls]['ap'], 2) for cls in AP_sequence.keys()}
             ))
    print(f"Average mAP of sequence: {round( np.nanmean([val['ap'] for val in AP_sequence.values()]) , 2)}")

    return AP_sequence, tp_data_seq

