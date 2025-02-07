import numpy as np
from typing import Union, Literal, List
import json

def bbox2d_area(pos_xywh:List[float]) -> float:
    """Calculate the area of Bounding Box

    Args:
        pos_xyhw (List[float]): Coordinates of 2D bounding boxes in x, y, width, height format

    Returns:
        float: Area of bounding box
    """
    cx, cy, h, w =  pos_xywh
    return h*w

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
