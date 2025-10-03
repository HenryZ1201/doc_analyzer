from pathlib import Path
from loguru import logger
from huggingface_hub import snapshot_download
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
import yaml
import os
# import pdf_extract_kit.tasks  # Ensure the task and model are registered
import pdf_extract_kit.tasks.layout_detection.task
import pdf_extract_kit.tasks.layout_detection.models.yolo  # Ensure the task and model are registered
from pdf_extract_kit.utils.table_processor import process_latex_table
from pdf_extract_kit.utils.json_processor import save_list_of_dicts_to_json
from pdf_extract_kit.utils.table_processor import yolo_to_pdf_bbox, visualize_and_save_pdf_bbox, extract_text_from_pdf_bbox, process_markdown_table

import fitz  # PyMuPDF
from PIL import Image
import pdfplumber
import cv2
import numpy as np
# from pdf_extract_kit.utils.table_processor import visualize_pdf_bbox


current_dir = Path(__file__).resolve().parent




if __name__ == "__main__":
    # doc_path = 'source/漳州统计年鉴2024_1-56 copy.pdf'
    # doc_path = 'source/漳州统计年鉴2024_483-520.pdf'
    # doc_path = 'source/漳州统计年鉴2024_19-52.pdf'
    # doc_path = 'source/漳州统计年鉴2024_19-52 copy.pdf'
    # doc_path = 'source/漳州统计年鉴2024_20.pdf'
    doc_path = 'source/漳州统计年鉴—2024_Ori_23-59.pdf'

    # Download the model files from the Hugging Face repository, if not already present
    model_save_dir = current_dir / 'models' / 'Layout' / 'YOLO'
    # model_name = 'doclayout_yolo_ft.pt'
    model_name = 'doclayout_yolo_docstructbench_imgsz1280_2501.pt'
    if not (model_save_dir / model_name).exists():
        # remind the user to download the model file
        logger.info(f"Model file {model_name} does not exist in {model_save_dir}. Downloading from Hugging Face...")
        snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir=str(current_dir), allow_patterns='models/Layout/YOLO/doclayout_yolo_ft.pt')
    else:
        logger.info(f"✅ Model file {model_name} exist in {model_save_dir}.")


    doc_name = Path(doc_path).stem
    layout_config = {
        'inputs': doc_path,
        'outputs': f'output/layout_detection/{doc_name}',
        'tasks': {
            'layout_detection': {
                'model': 'layout_detection_yolo',
                'model_config': {
                    'model_path': str(model_save_dir / model_name),
                    'img_size': 1280,   # 1024 for doclayout_yolo_ft.pt, 1280 for doclayout_yolo_docstructbench_imgsz1280_2501.pt
                    'conf_thres': 0.25,
                    'iou_thres': 0.45,
                    'visualize': True,  # whether to save visualization images
                    'nc': 10,
                    'workers': 8,
                    # 'device': 'cuda',
                    'device': 'mps' 
                },
                'save_table_images': True,  # whether to save cropped table images
                'table_margin': 0   # margin to add when cropping table images
            }
        }
    }
    
    print(yaml.dump(layout_config, allow_unicode=True, sort_keys=False))

    task_instances = initialize_tasks_and_models(layout_config)

    # Layout detection task
    model_layout_detection = task_instances.get('layout_detection')

    # for image detection
    input_data = layout_config['inputs']
    result_path = layout_config['outputs']

    # for pdf detection
    if not os.path.exists(os.path.join(result_path, 'document_tables_info.json')): 
        logger.info(f"Result will be saved to {result_path}")
        detection_results, doc_tables_info_list = model_layout_detection.predict_pdfs(input_data, result_path)
        pdf_images_info = model_layout_detection.get_pdf_images_info()
        save_list_of_dicts_to_json(pdf_images_info, os.path.join(result_path, 'pdf_images_info.json'))
        pass
    else:
        logger.info(f"✅ Layout detection results already exist in {result_path}/document_tables_info.json. Skipping layout detection step.")
        # load the existing document_tables_info.json file
        with open(os.path.join(result_path, 'document_tables_info.json'), 'r', encoding='utf-8') as f:
            doc_tables_info_list = yaml.safe_load(f)
        with open(os.path.join(result_path, 'pdf_images_info.json'), 'r', encoding='utf-8') as f:
            pdf_images_info = yaml.safe_load(f)

    """
    Table recognition task
    Loop over the detected tables, transform the yolo-based coordinates to pdfplumber-based coordinates,
    and then perform table recognition using the OCR-based method.
    """
    for idx, table_info in enumerate(doc_tables_info_list):
        # obtain table info
        table_image_path = table_info['table_image_path']
        table_id = table_info['table_id']
        page_index = table_info['page_index']
        logger.info(f"Processing table {idx+1}/{len(doc_tables_info_list)}: Table ID {table_id} on Page {page_index}, Image Path: {table_image_path}")

        # use function to convert yolo-based coordinates to pdfplumber-based coordinates, input is pdf_path, page_index, yolo_box
        img_id = f"{Path(doc_path).stem}_page_{page_index:04d}"
        yolo_bbox = table_info['bbox']
        image_width = pdf_images_info[img_id]['image_width']
        image_height = pdf_images_info[img_id]['image_height']
        dpi = pdf_images_info[img_id]['dpi']

        
        # load the pdf page
        doc = fitz.open(doc_path)
        page = doc[page_index-1]  # page_index starts from 1
        # get pdf page properties
        pdf_width, pdf_height = page.mediabox.width, page.mediabox.height

        # convert yolo bbox to pdf bbox
        pdf_bbox = yolo_to_pdf_bbox(yolo_bbox, image_width, image_height, pdf_width, pdf_height)
        logger.info(f"Table ID {table_id} on Page {page_index}: YOLO bbox {yolo_bbox} converted to PDF bbox {pdf_bbox} (in points)")

        # visualize the bbox on the pdf page and save the result
        pdf_rectangle_save_path = os.path.join(result_path, 'pdf_with_detected_table_bbox')
        if not os.path.exists(pdf_rectangle_save_path):
            os.makedirs(pdf_rectangle_save_path)
        visualize_and_save_pdf_bbox(doc, page_index-1, pdf_bbox,
                           output_pdf_path=os.path.join(pdf_rectangle_save_path, f"{img_id}_with_table_{table_id}_bbox.pdf"))
        plain_text_table, markdown_table = extract_text_from_pdf_bbox(doc, page_index-1, pdf_bbox)
        logger.info(f"Extracted text from Table ID {table_id} on Page {page_index}:\n{plain_text_table}")
        logger.info(f"Extracted text from Table ID {table_id} on Page {page_index}:\n{markdown_table}")
        # save the markdown table to a .md file
        md_path = os.path.join(result_path, f"{img_id}_table_{table_id:04d}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_table)
            logger.info(f"Saved extracted markdown table to {md_path}") 
        # process_markdown_table(md_path)
        pass




        # # *** MODIFICATION 2: Updated function signature ***
        # # pdf_bbox = convert_yolo_to_pdf_bbox(yolo_bbox, image_width, image_height, page_rect, dpi)
        # pdf_bbox = convert_yolo_to_pdf_bbox(yolo_bbox, image_width, image_height, pdf_width, pdf_height, dpi)
        # logger.info(f"Table ID {table_id} on Page {page_index}: YOLO bbox {yolo_bbox} converted to PDF bbox {pdf_bbox} (in points)")

        # # pdf_bbox = convert_yolo_to_pdf_bbox(yolo_bbox, image_width, image_height, pdf_width, pdf_height, dpi)
        # # logger.info(f"Table ID {table_id} on Page {page_index}: YOLO bbox {yolo_bbox} converted to PDF bbox {pdf_bbox} (in points)")
        # # visualize the bbox on the pdf page and save the result
        # pdf_rectangle_save_path = os.path.join(result_path, 'pdf_with_detected_table_bbox')
        # if not os.path.exists(pdf_rectangle_save_path):
        #     os.makedirs(pdf_rectangle_save_path)
        # # *** MODIFICATION 3: Passing YOLO image dimensions to visualization function ***
        # visualize_pdf_bbox(doc_path, page_index-1, pdf_bbox, dpi=144, 
        #                    output_pdf=os.path.join(pdf_rectangle_save_path, f"{img_id}_with_table_{table_id}_bbox.pdf"),
        #                    output_image=os.path.join(pdf_rectangle_save_path, f"{img_id}_with_table_{table_id}_bbox.png"))

        # pass
 

    
    

