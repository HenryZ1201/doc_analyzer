import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PIL import Image
import logging
from pathlib import Path


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def download_unitable_model(model_name, cache_dir):
    """Download UniTable model to a specified directory."""
    try:
        logging.info(f"Downloading model {model_name} to {cache_dir}")
        snapshot_download(
            repo_id=model_name,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            cache_dir=cache_dir
        )
        logging.info(f"Model downloaded successfully to {cache_dir}")
    except Exception as e:
        logging.error(f"Failed to download model: {str(e)}")
        raise

def load_unitable_model(model_path):
    """Load UniTable model and tokenizer from local directory."""
    try:
        logging.info(f"Loading model from {model_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        logging.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise

def image_to_latex(image_path, model, tokenizer, output_format="latex"):
    """Perform table recognition and return LaTeX output."""
    try:
        # Load and preprocess image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        logging.info(f"Processing image: {image_path}")
        image = Image.open(image_path).convert("RGB")

        # Dummy implementation for UniTable inference (replace with actual UniTable API)
        # Note: UniTable's exact inference API may vary; check repo for specifics
        # This assumes a simplified seq2seq generation for structure
        inputs = tokenizer(image, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs, max_length=4096)
        table_structure = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Convert to LaTeX (simplified; adapt based on UniTable's output format)
        # Assuming table_structure is HTML-like or JSON; convert to LaTeX
        latex_output = convert_to_latex(table_structure)
        logging.info("Table recognition completed")
        return latex_output
    except Exception as e:
        logging.error(f"Table recognition failed: {str(e)}")
        raise

def convert_to_latex(structure):
    """Convert UniTable output to LaTeX format with longtable support."""
    # Placeholder: Actual conversion depends on UniTable's output (e.g., HTML/JSON)
    # This is a simplified example assuming structure is a list of rows
    try:
        latex = "\\documentclass{article}\n\\usepackage{longtable}\n\\begin{document}\n"
        latex += "\\begin{longtable}{|c|c|c|}\n\\hline\n"
        latex += "Header1 & Header2 & Header3 \\\\\n\\hline\n"
        # Example: Parse structure into rows and cells
        # Replace with actual parsing logic based on UniTable output
        for row in structure.get("rows", []):  # Dummy structure
            cells = row.get("cells", [])
            latex += " & ".join(cells) + " \\\\\n\\hline\n"
        latex += "\\end{longtable}\n\\end{document}"
        return latex
    except Exception as e:
        logging.error(f"LaTeX conversion failed: {str(e)}")
        raise

def save_latex_to_file(latex_content, output_file="latex.txt"):
    """Save LaTeX content to a file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_content)
        logging.info(f"LaTeX output saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save LaTeX file: {str(e)}")
        raise

def main():
    # Configuration
    current_dir = Path(__file__).resolve().parent

    model_name = "poloclub/unitable-large-structure"  # UniTable model for structure
    cache_dir = current_dir / 'UniTable_model'  # Specify your target directory
    image_path = "D:\\Codes\\2025\\RAG\doc_analyzer\\output\\layout_detection\\漳州统计年鉴2024_19-52\\table_images\\漳州统计年鉴2024_19-52_page_0002_table_0001.png"  # Your table image
    output_file = "latex.txt"

    # Step 1: Download model
    download_unitable_model(model_name, cache_dir)

    # Step 2: Load model
    model, tokenizer = load_unitable_model(cache_dir)

    # Step 3: Perform table recognition
    latex_output = image_to_latex(image_path, model, tokenizer, output_format="latex")

    # Step 4: Save LaTeX to file
    save_latex_to_file(latex_output, output_file)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Script completed successfully")

if __name__ == "__main__":
    main()