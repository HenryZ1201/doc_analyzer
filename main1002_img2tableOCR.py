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

from loguru import logger
from img2table.ocr import PaddleOCR, TesseractOCR
from img2table.document import Image

import pandas as pd
from typing import Union

def post_process_table_sheets(xlsx_path: Union[str, Path]):
    """
    Analyzes all sheets in an XLSX file and concatenates them either vertically 
    (if columns match) or horizontally (if rows match). The original file is 
    then overwritten with the single stacked sheet.

    NOTE: Native Excel formatting (like merged cell properties, colors, etc.) 
    cannot be preserved using this pandas-based stacking approach. However, 
    the content of merged cells is preserved structurally using forward fill.

    Args:
        xlsx_path: Path to the input Excel file.
    """
    input_path = Path(xlsx_path)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Loading sheets from: {input_path.name}")
    
    # Load all sheets into a dictionary of DataFrames. sheet_name=None reads all.
    try:
        # Load with header=None to treat all rows as data for proper stacking
        all_dfs = pd.read_excel(input_path, sheet_name=None, header=None)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    df_list = list(all_dfs.values())
    num_sheets = len(df_list)

    if num_sheets <= 1:
        print("File contains 0 or 1 sheet. No stacking necessary.")
        return

    # --- 1. Check for Vertical Concatenation (Same number of columns) ---
    col_counts = [df.shape[1] for df in df_list]
    
    # Check if all column counts are identical
    if len(set(col_counts)) == 1:
        print(f"All {num_sheets} sheets have {col_counts[0]} columns. Performing vertical stacking (rows)...")
        
        # For vertical stacking (e.g., header + data):
        # We process each sheet to fill NaNs created by merged cells horizontally 
        # (ffill(axis=1)), ensuring header content is propagated correctly before stacking.
        processed_dfs = []
        for df in df_list:
            # Fills NaNs in a row with the last valid non-NaN value in that row.
            # This preserves the content structure of horizontally merged cells.
            processed_dfs.append(df.ffill(axis=1)) 
            
        # Concatenate all processed sheets vertically.
        final_df = pd.concat(processed_dfs, axis=0, ignore_index=True)

        
        concat_axis = "Vertical (Rows)"
    
    # --- 2. Check for Horizontal Concatenation (Same number of rows) ---
    else:
        row_counts = [df.shape[0] for df in df_list]
        
        # Check if all row counts are identical
        if len(set(row_counts)) == 1:
            print(f"All {num_sheets} sheets have {row_counts[0]} rows. Performing horizontal stacking (columns)...")
            
            # Concatenate along axis=1 (columns)
            final_df = pd.concat(df_list, axis=1)
            concat_axis = "Horizontal (Columns)"
        else:
            print("Sheets cannot be stacked: Column counts are not equal, and row counts are not equal.")
            return

    # --- 3. Save the Result (Overwrite Original File) ---
    
    # Use the original input path as the output path
    output_path = input_path
    
    print(f"Saving concatenated table, replacing the original file: {output_path.name}")

    # Use ExcelWriter to ensure the file is completely overwritten and only 
    # the single DataFrame is written as 'Sheet1'.
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # header=False is used since we read the sheets without headers
            final_df.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
    except Exception as e:
        print(f"Error saving to Excel file: {e}")
        return
    
    print(f"\n✅ Success: Sheets were stacked {concat_axis} and the original file was replaced at {output_path}")

# Configure logging (assuming logger is already set up elsewhere)
current_dir = Path(__file__).resolve().parent

if __name__ == "__main__":
    # doc_path = 'source/漳州统计年鉴2024_1-56 copy.pdf'
    # doc_path = 'source/漳州统计年鉴2024_483-520.pdf'
    doc_path = 'source/漳州统计年鉴2024_19-52.pdf'
    # doc_path = 'source/漳州统计年鉴2024_20.pdf'


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

    # Load configuration
    # config_path = current_dir / 'configs' / 'layout_detection.yaml'
    # layout_config = load_config(config_path)


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
                    'device': 'mps',    #'cuda'
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

    logger.info(f"Result will be saved to {result_path}")

    if not os.path.exists(os.path.join(result_path, 'document_tables_info.json')): 
        detection_results, doc_tables_info_list = model_layout_detection.predict_pdfs(input_data, result_path)
    else:
        logger.info(f"✅ Layout detection results already exist in {result_path}/document_tables_info.json. Skipping layout detection step.")
        # load the existing document_tables_info.json file
        with open(os.path.join(result_path, 'document_tables_info.json'), 'r', encoding='utf-8') as f:
            doc_tables_info_list = yaml.safe_load(f)


    # start table parsing
    # table_output_format = 'markdown'  # 'latex', 'markdown', 'html'
    # table_output_dir = f'output/layout_detection/{doc_name}/table_images'
    # MODEL_DIR = os.path.expanduser('/models/paddleocr')
    
    # # With kw for custom options
    # ocr = PaddleOCR(
    #     lang="ch",
    #     kw={
    #         'use_doc_orientation_classify': False,      # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    #         'use_doc_unwarping': False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    #         'use_textline_orientation': False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
    #         # 'ocr_dir': MODEL_DIR, # <-- This is the key fix
    #         'ocr_version': 'PP-OCRv5',
    #         # 'use_gpu': False,              # Force CPU usage
    #         'cpu_threads': 8,              # Set CPU thread count

    #         'det_limit_side_len': 1024,
    #         'det_limit_type': 'max', 

    #         # # Adjust sensitivity for text areas (optional, but recommended for faint/dense text)
    #         # 'det_db_box_thresh': 0.5,
    #         # 'det_db_unclip_ratio': 1.0, # Prevents bounding boxes from merging
    #     }
    # )

    # for idx, table_info in enumerate(doc_tables_info_list):
    #     # obtain table info
    #     table_image_path = table_info['table_image_path']
    #     table_id = table_info['table_id']
    #     page_index = table_info['page_index']
    #     logger.info(f"Processing table {idx+1}/{len(doc_tables_info_list)}: Table ID {table_id} on Page {page_index}, Image Path: {table_image_path}")
        
    #     image_name = os.path.basename(table_image_path).split('.')[0]
        
    #     output_xlsx_path = os.path.join(table_output_dir, f"{image_name}_TesseractOCR.xlsx")
    #     doc = Image(src=table_image_path)
    #     extracted_tables = doc.to_xlsx(
    #         dest=output_xlsx_path,
    #         ocr=ocr,
            
    #         # 🌟 CRITICAL: Tell img2table to process the image for borderless table structures
    #         borderless_tables=True,
            
    #         # Optional: These flags can help structure the table data more accurately
    #         implicit_rows=True,
    #         implicit_columns=True,
            
    #         # Confidence threshold (optional, adjust if too much noise is extracted)
    #         min_confidence=30,

    #     )    
    #     post_process_table_sheets(output_xlsx_path)
        

    # pass

    for idx, table_info in enumerate(doc_tables_info_list):
        # obtain table info
        table_image_path = table_info['table_image_path']
        table_id = table_info['table_id']
        page_index = table_info['page_index']
        logger.info(f"Processing table {idx+1}/{len(doc_tables_info_list)}: Table ID {table_id} on Page {page_index}, Image Path: {table_image_path}")
        image_name = os.path.basename(table_image_path).split('.')[0]
        output_xlsx_path = os.path.join(result_path, f"table_images/{image_name}_TesseractOCR.xlsx")

        ocr = TesseractOCR(n_threads=8, lang="chi_sim")
        doc = Image(src=table_image_path)
        extracted_tables = doc.to_xlsx(
            ocr=ocr,
            implicit_columns=True,
            implicit_rows=True,
            borderless_tables=True,
            min_confidence=50,
            dest=output_xlsx_path
        )
        post_process_table_sheets(output_xlsx_path)

    
    

