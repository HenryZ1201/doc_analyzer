import os
import cv2
import torch
from pdf_extract_kit.registry import MODEL_REGISTRY
from pdf_extract_kit.utils.visualization import visualize_bbox
from loguru import logger
import numpy as np
import json
from pdf_extract_kit.utils.json_processor import save_list_of_dicts_to_json
# from pdf_extract_kit.dataset.dataset import ImageDataset
                
@MODEL_REGISTRY.register('layout_detection_yolo')
class LayoutDetectionYOLO:
    def __init__(self, config):
        """
        Initialize the LayoutDetectionYOLO class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.save_table_images = config.get('save_table_images', True)
        self.table_margin = config.get('table_margin', 30)  # margin to add when cropping table images
        # Mapping from class IDs to class names

        # for yolo original
        # self.id_to_names = {
        #     0: 'title', 
        #     1: 'plain text',
        #     2: 'abandon', 
        #     3: 'figure', 
        #     4: 'figure_caption', 
        #     5: 'table', 
        #     6: 'table_caption', 
        #     7: 'table_footnote', 
        #     8: 'isolate_formula', 
        #     9: 'formula_caption'
        # }

        # for better yolo
        self.id_to_names = {
            0: 'title', 
            1: 'plain text',
            2: 'abandon', 
            3: 'figure', 
            4: 'figure_caption', 
            5: 'table', 
            6: 'table_caption', 
            7: 'table_footnote', 
            8: 'interline_equation', 
            9: 'interline_equation_number',
            13: 'inline_equation',
            14: 'interline_equation_yolo',
            15: 'ocr_text',
            16: 'low_score_text',
            101: 'image_footnote',
        }

        # Load the YOLO model from the specified path
        try:
            from doclayout_yolo import YOLOv10
            self.model = YOLOv10(config['model_path'])
        except AttributeError:
            from ultralytics import YOLO
            self.model = YOLO(config['model_path'])

        # Set model parameters
        self.img_size = config.get('img_size', 1280)
        self.conf_thres = config.get('conf_thres', 0.25)
        self.iou_thres = config.get('iou_thres', 0.45)
        self.visualize = config.get('visualize', False)
        self.nc = config.get('nc', 10)
        self.workers = config.get('workers', 8)
        self.device = config.get('device', 'cpu')
        
        if self.iou_thres > 0:
            import torchvision
            self.nms_func = torchvision.ops.nms

    def predict(self, images, result_path, image_ids=None):
        """
        Predict formulas in images.

        Args:
            images (list): List of images to be predicted.
            result_path (str): Path to save the prediction results.
            image_ids (list, optional): List of image IDs corresponding to the images.

        Returns:
            list: List of prediction results.
        """
        results = []

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        table_images_path = os.path.join(result_path, 'table_images')
        if self.save_table_images and not os.path.exists(table_images_path):
            os.makedirs(table_images_path)
            do_save_table_images = True
        else:
            logger.info(f"✅ Table images path {table_images_path} already exists. Will skip saving table images.")
            do_save_table_images = False

        layout_save_path = os.path.join(result_path, 'layout_visualizations')
        if self.visualize and not os.path.exists(layout_save_path):
            os.makedirs(layout_save_path)
            do_save_layout_visualization = True
        else:
            logger.info(f"✅ Layout visualization path {layout_save_path} already exists. Will skip saving layout visualization.")
            do_save_layout_visualization = False    

        page_table_info_list = list()
        for idx, image in enumerate(images):
            logger.info(f"Processing page image {idx+1}/{len(images)}")
            result = self.model.predict(image, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, verbose=False, device=self.device)[0]
            
            # Determine the base name of the image
            if image_ids:
                base_name = image_ids[idx]
            else:
                # base_name = os.path.basename(image)
                base_name = os.path.splitext(os.path.basename(image))[0]  # Remove file extension
            
            # determine boxes, classes, scores
            boxes = result.__dict__['boxes'].xyxy
            classes = result.__dict__['boxes'].cls
            scores = result.__dict__['boxes'].conf

            if self.iou_thres > 0:
                indices = self.nms_func(boxes=torch.Tensor(boxes), scores=torch.Tensor(scores),iou_threshold=self.iou_thres)
                boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
                if len(boxes.shape) == 1:
                    boxes = np.expand_dims(boxes, 0)
                    scores = np.expand_dims(scores, 0)
                    classes = np.expand_dims(classes, 0)

            # save table images
            if do_save_table_images:
                table_info_list = self.save_table_info(idx, image, table_images_path, classes, boxes, base_name, table_margin=self.table_margin)
                page_table_info_list.extend(table_info_list)
            # save layout visualization
            if do_save_layout_visualization:
                vis_result = visualize_bbox(image, boxes, classes, scores, self.id_to_names)
                result_name = f"{base_name}_layout.png"
                cv2.imwrite(os.path.join(layout_save_path, result_name), vis_result)
            results.append(result)
    
        
        save_list_of_dicts_to_json(page_table_info_list, os.path.join(result_path, 'document_tables_info.json'))  # save all pages table info to a json file

        return results, page_table_info_list
    

    @staticmethod
    def save_table_info(idx, image, table_images_path, classes, boxes, base_name, table_margin):
        """
        idx: index of the image in the batch, starting from 0
        image: PIL.Image object
        table_images_path: path to save the cropped table images
        classes: list of class ids
        boxes: list of bounding boxes, each represented as [x_min, y_min, x_max, y_max]
        base_name: base name of the image (without extension)   
        table_margin: margin to add when cropping table images

        Return table_info: list of dict, each dict contains:
            - bbox: [x_min, y_min, x_max, y_max]
            - table_image_path: path to the cropped table image
            - page_index: index of the page (starting from 1)
            - table_id: id of the table on the page (starting from 1)
        """
        table_count = 0
        table_info_list = list()
        for i, cls in enumerate(classes):
            cls = int(cls)
            if cls == 5:  # Table class
                
                box = np.round(boxes[i].cpu().numpy()).astype(int)  # 先四舍五入再转int
                x1, y1, x2, y2 = box

                # 向外扩展 table_margin
                x1 = max(0, x1 - table_margin)
                x2 = x2 + table_margin
                y1 = max(0, y1 - table_margin)
                y2 = y2 + table_margin
                
                # table name
                table_img_name = f"{base_name}_table_{table_count+1:04d}.png"

                # table image data
                table_img = image.crop((x1, y1, x2, y2))
                table_img_np = np.array(table_img)  # PIL.Image -> numpy array (RGB)
                table_img_bgr = cv2.cvtColor(table_img_np, cv2.COLOR_RGB2BGR)  # RGB -> BGR
                cv2.imwrite(os.path.join(table_images_path, table_img_name), table_img_bgr)
                logger.info(f"Saved table image: {table_img_name}")

                # table info
                table_info = {
                    'page_index': idx + 1,  # page number starting from 1
                    'table_id': table_count + 1,  # table body id on the page, starting from 1   
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'table_image_path': os.path.join(table_images_path, table_img_name)
                }
                table_info_list.append(table_info)
                table_count += 1
        
        return table_info_list
    
    # @staticmethod
    # def save_list_of_dicts_to_json(data, file_path):
    #     """
    #     Save a list of dictionaries to a JSON file.

    #     Args:
    #         data (list): List of dictionaries.
    #         file_path (str): Path to the output JSON file.
    #     """
    #     with open(file_path, 'w', encoding='utf-8') as f:
    #         json.dump(data, f, ensure_ascii=False, indent=4)