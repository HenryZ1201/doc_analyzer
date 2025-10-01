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


current_dir = Path(__file__).resolve().parent

if __name__ == "__main__":
    # doc_path = 'source/漳州统计年鉴2024_1-56 copy.pdf'
    # doc_path = 'source/漳州统计年鉴2024_483-520.pdf'
    doc_path = 'source/漳州统计年鉴2024_19-52.pdf'


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
                    'device': 'cpu'
                },
                'save_table_images': True,  # whether to save cropped table images
                'table_margin': 30   # margin to add when cropping table images
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
    table_output_format = 'latex'  # 'latex', 'markdown', 'html'
    table_output_dir = f'output/layout_detection/{doc_name}/table_images'
    table_parsing_config = {
        'inputs': None,
        'outputs': table_output_dir,
        'tasks': {
            'table_parsing': {
                'model': 'table_parsing_struct_eqtable',
                'model_config': {
                    'model_path': 'models/TabRec/StructEqTable',
                    'max_new_tokens': 4096,
                    'max_time': 30,
                    'output_format': table_output_format,
                    'lmdeploy': False,
                    'flash_atten': True,
                }
            }
        }
    }
    snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir=str(current_dir), allow_patterns='models/TabRec/StructEqTable/*')
    table_model_save_dir = current_dir / 'models' / 'TabRec' / 'StructEqTable'
    # model_name = 'doclayout_yolo_ft.pt'
    if not any(table_model_save_dir.iterdir()):        # remind the user to download the model file
        # remind the user to download the model file
        logger.info(f"Table Model file {table_model_save_dir} does not exist in. Downloading from Hugging Face...")
        snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir=str(current_dir), allow_patterns='models/TabRec/TabRec/*')
    else:
        logger.info(f"✅ Model file {table_model_save_dir} exist.")

    print(yaml.dump(table_parsing_config, allow_unicode=True, sort_keys=False))    
    
    for idx, table_info in enumerate(doc_tables_info_list):
        # obtain table info
        table_image_path = table_info['table_image_path']
        table_id = table_info['table_id']
        page_index = table_info['page_index']
        logger.info(f"Processing table {idx+1}/{len(doc_tables_info_list)}: Table ID {table_id} on Page {page_index}, Image Path: {table_image_path}")

        # specify the task
        table_parsing_config['inputs'] = table_image_path
        task_instances = initialize_tasks_and_models(table_parsing_config)
        model_table_parsing = task_instances.get('table_parsing')

        # infer on the table image
        parsing_results = model_table_parsing.predict(table_image_path, table_output_dir, output_format=table_output_format)
        # save the parsing results
        parsing_result_path = Path(table_output_dir) / f"{doc_name}_page_{page_index:04d}_table_{table_id:04d}_{table_output_format}.txt"
        with open(parsing_result_path, 'w', encoding='utf-8') as f:
            f.write(parsing_results[0])
        logger.info(f"Saved parsing result to {parsing_result_path}")
        # do the rendering and col&row counting
        rendered_table_path, num_cols, num_rows = process_latex_table(parsing_result_path)

        # save to tabel info
        table_info['parsed_content'] = parsing_results[0]
        table_info['parsed_path'] = str(parsing_result_path)
        table_info['rendered_path'] = str(rendered_table_path)
        table_info['estimated_ncols'] = num_cols
        table_info['estimated_nrows'] = num_rows

    document_tables_parsed_info_dir = os.path.join(result_path, 'document_tables_parsed_info.json')
    save_list_of_dicts_to_json(doc_tables_info_list, document_tables_parsed_info_dir)

    pass


    
    

