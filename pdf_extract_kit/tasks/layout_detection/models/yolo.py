import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from pdf_extract_kit.registry import MODEL_REGISTRY
from pdf_extract_kit.utils.visualization import visualize_bbox
from pdf_extract_kit.dataset.dataset import ImageDataset
import torchvision.transforms as transforms


import numpy as np

def iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two boxes.

    Args:
        box1 (list or np.array): A bounding box (x1, y1, x2, y2).
        box2 (list or np.array): A bounding box (x1, y1, x2, y2).

    Returns:
        float: IoU between box1 and box2.
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # IoU
    return inter_area / union_area if union_area > 0 else 0

def filter_boxes(boxes, classes, scores, iou_threshold=0.95):
    """
    Filter boxes to keep only the larger ones when they overlap by more than IoU threshold.

    Args:
        boxes (torch.Tensor): A tensor of bounding boxes (x1, y1, x2, y2).
        classes (torch.Tensor): A tensor of class IDs corresponding to the boxes.
        scores (torch.Tensor): A tensor of scores corresponding to the boxes.
        iou_threshold (float): Threshold for IoU to consider overlap (default 0.95).

    Returns:
        tuple: Filtered bounding boxes, classes, and scores.
    """
    filtered_boxes = []
    filtered_classes = []
    filtered_scores = []

    keep = [True] * len(boxes)

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        box_i = boxes[i]
        for j in range(len(boxes)):
            if i != j and keep[j]:
                box_j = boxes[j]
                overlap = iou(box_i, box_j)
                if overlap > iou_threshold:
                    # If boxes overlap more than the threshold, keep the larger box
                    area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                    area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                    if area_i > area_j:
                        keep[j] = False  # Remove the smaller box
                    else:
                        keep[i] = False  # Remove the current box
                        break

    for i in range(len(boxes)):
        if keep[i]:
            filtered_boxes.append(boxes[i].cpu().numpy())  # Move to CPU before converting to NumPy
            filtered_classes.append(classes[i].cpu().numpy())  # Move to CPU
            filtered_scores.append(scores[i].cpu().numpy())  # Move to CPU

    # Convert lists back to numpy arrays
    filtered_boxes = np.array(filtered_boxes)
    filtered_classes = np.array(filtered_classes)
    filtered_scores = np.array(filtered_scores)

    return filtered_boxes, filtered_classes, filtered_scores


@MODEL_REGISTRY.register('layout_detection_yolo')
class LayoutDetectionYOLO:
    def __init__(self, config):
        """
        Initialize the LayoutDetectionYOLO class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        # Mapping from class IDs to class names
        self.id_to_names = {
            0: 'title',
            1: 'plain text',
            2: 'abandon',
            3: 'figure',
            4: 'figure_caption',
            5: 'table',
            6: 'table_caption',
            7: 'table_footnote',
            8: 'isolate_formula',
            9: 'formula_caption'
        }

        # Load the YOLO model from the specified path
        self.model = YOLO(config['model_path'])

        # Set model parameters
        self.img_size = config.get('img_size', 1280)
        self.pdf_dpi = config.get('pdf_dpi', 200)
        self.conf_thres = config.get('conf_thres', 0.25)
        self.iou_thres = config.get('iou_thres', 0.45)
        self.visualize = config.get('visualize', False)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.get('batch_size', 1)
        self.max_det = config.get('max_det', 300)
        self.nc = config.get('nc', 10)
        self.workers = config.get('workers', 8)

    def predict(self, images, result_path, image_ids=None):
        """
        Predict formulas in images, visualize detections, and save results.

        Args:
            images (list of str): Paths to the input images.
            result_path (str): Directory where prediction results and visualizations are saved.
            image_ids (list of str, optional): IDs corresponding to the images.

        Returns:
            list: Prediction results for each image.

        Output:
            - Cropped images of detected formulas are saved in 'result_path/img/'.
            - Visualized images with bounding boxes are saved directly in 'result_path/'.
        """
        results = []
        for idx, image_path in enumerate(images):
            result = self.model.predict(image_path, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, verbose=False)[0]
            if self.visualize:
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                boxes = result.__dict__['boxes'].xyxy
                classes = result.__dict__['boxes'].cls
                scores = result.__dict__['boxes'].conf

                boxes, classes, scores = filter_boxes(boxes, classes, scores)
                # sort boxes by their y1 coordinate and x1 coordinate
                sorted_indices = boxes[:, 1].argsort()
                boxes = boxes[sorted_indices]

                classes = classes[sorted_indices]
                scores = scores[sorted_indices]

                # Visualize the bounding boxes on the image separately

                # Extract the image name without extension
                image_name = os.path.splitext(os.path.basename(image_path))[0]

                # Save in subfolder
                save_dir = os.path.join(result_path, "img")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Visualize and export the bounding boxes on the image separately
                for i, box in enumerate(boxes):
                    image = cv2.imread(image_path)
                    x1, y1, x2, y2 = map(int, box)
                    #print(image_path)
                    #print("Box:", i, "Coordinates:", x1, y1, x2, y2)
                    bbox_image = image[y1:y2, x1:x2]

                    class_id = int(classes[i])
                    class_name = self.id_to_names.get(class_id, 'unknown')

                    # Construct the file path
                    file_path = os.path.join(save_dir, f"{i}_{class_name}.png")

                    # Save the cropped bounding box image
                    cv2.imwrite(file_path, bbox_image)

                # Visualize the bounding boxes on the image
                vis_result = visualize_bbox(image_path, boxes, classes, scores, self.id_to_names)

                # Determine the base name of the image
                if image_ids:
                    base_name = image_ids[idx]
                else:
                    # base_name = os.path.basename(image)
                    base_name = os.path.splitext(os.path.basename(image_path))[0]  # Remove file extension

                result_name = f"{base_name}_MFD.png"

                # Save the visualized result
                cv2.imwrite(os.path.join(result_path, result_name), vis_result)
            results.append(result)
        return results