import os
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Tuple
from collections import Counter
import fitz
from pdf_extract_kit.utils.data_preprocess import load_pdf_page
from loguru import logger
import numpy as np
from collections import Counter


# Set Chinese font globally for Matplotlib to display Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False # Correctly display minus signs

def yolo_to_pdf_bbox(
    yolo_bbox: list,
    image_width: int,
    image_height: int,
    pdf_width: float,
    pdf_height: float
) -> tuple[float, float, float, float]:
    """
    Converts a bounding box from image-pixel coordinates (e.g., from YOLO detection
    on a PDF-derived image) to PDF point coordinates.

    This function calculates the scale factor directly from the known pixel dimensions
    of the rendered image and the point dimensions of the original PDF page.

    Args:
        yolo_bbox: Bounding box in pixel coordinates [x_min, y_min, x_max, y_max].
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.
        pdf_width: Width of the PDF page in points.
        pdf_height: Height of the PDF page in points.

    Returns:
        A tuple (x0, y0, x1, y1) representing the bounding box in PDF points.
    """
    if image_width == 0 or image_height == 0:
        logger.error("Image dimensions cannot be zero for conversion.")
        return (0.0, 0.0, 0.0, 0.0)

    # 1. Calculate scaling factors (points per pixel)
    # Scale_x = Image_Width / PDF_Width
    scale_x = image_width / pdf_width
    scale_y = image_height / pdf_height

    # 2. Convert pixel coordinates back to point coordinates
    x_min_pdf = yolo_bbox[0] / scale_x
    y_min_pdf = yolo_bbox[1] / scale_y
    x_max_pdf = yolo_bbox[2] / scale_x
    y_max_pdf = yolo_bbox[3] / scale_y

    logger.info(
        f"Conversion: YOLO BBox {yolo_bbox} (Pixels) -> PDF BBox "
        f"({x_min_pdf:.2f}, {y_min_pdf:.2f}, {x_max_pdf:.2f}, {y_max_pdf:.2f}) (Points)"
    )

    return (x_min_pdf, y_min_pdf, x_max_pdf, y_max_pdf)


def visualize_and_save_pdf_bbox(
    doc: fitz.Document,
    page_index: int,
    pdf_bbox: tuple[float, float, float, float],
    output_pdf_path: str,
    color: tuple[float, float, float] = (0, 1, 0),  # Green in RGB [0, 1]
    width: float = 2.0
) -> None:
    """
    Draws a rectangle on a specific PDF page and saves the modified document as a 
    new single-page PDF file.

    Args:
        doc: The open fitz.Document object.
        page_index: The 0-based index of the page to modify.
        pdf_bbox: The bounding box in PDF points (x0, y0, x1, y1).
        output_pdf_path: The path to save the modified single-page PDF file.
        color: RGB tuple for the rectangle color (e.g., (0, 1, 0) for green).
        width: The line width of the rectangle border in points.
    """
    try:
        page = doc[page_index]
    except IndexError:
        logger.error(f"Page index {page_index} is out of range for the document.")
        return

    rect = fitz.Rect(pdf_bbox)

    # Draw a rectangle with a border and no fill
    page.draw_rect(
        rect,
        color=color,        # Stroke color
        fill=None,          # No fill color
        width=width,        # Line width
        overlay=True        # Draw over existing content
    )
    logger.info(f"Drawn rectangle on Page {page_index + 1} at {rect}.")

    # --- Save only the specified single page PDF ---
    try:
        new_doc = fitz.open() # Create an empty document
        # Copy the modified page (page_index) from the source document (doc) to the new document.
        # This copies the page *with* the newly drawn rectangle.
        new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)

        # Save the new single-page document
        new_doc.save(output_pdf_path, garbage=4, deflate=True)
        new_doc.close()
        logger.info(f"Successfully saved single-page PDF to: {output_pdf_path}")
    except Exception as e:
        logger.error(f"Error saving single-page PDF: {e}")

def extract_text_from_pdf_bbox(
    doc: fitz.Document,
    page_index: int,
    pdf_bbox: tuple[float, float, float, float]
) -> tuple[str, str]:
    """
    Extracts a table from a specific bounding box on a PDF page.

    :param doc: The fitz.Document object.
    :param page_index: The 0-based index of the page.
    :param pdf_bbox: A tuple (x0, y0, x1, y1) representing the bounding box 
                     in PDF coordinates (points).
    :return: A tuple (plain_text_table, markdown_table) representing the 
             extracted table. Returns empty strings if no table is found.
    """
    if not 0 <= page_index < len(doc):
        return "Error: Page index out of range.", "Error: Page index out of range."

    page = doc[page_index]
    # Convert the input tuple to a fitz.Rect object for clipping
    clip_rect = fitz.Rect(pdf_bbox)

    # Use PyMuPDF's table extraction feature with the clipping rectangle
    # This automatically handles spatial relations, multi-line cells, and language.
    # It attempts to find tables *only* within the specified clip area.
    tabs = page.find_tables(clip=clip_rect)

    if not tabs.tables:
        # Fallback to simple text extraction if table detection fails, 
        # though this may not preserve table structure well.
        plain_text = page.get_text("text", clip=clip_rect)
        
        # Format the plain text slightly for the Markdown string for better clarity
        markdown_text = f"Could not find a structured table, raw text extracted:\n\n---\n\n{plain_text.strip()}"
        return plain_text, markdown_text

    # Assuming the content in the bbox is one table, take the first result
    table = tabs.tables[0]

    # Convert the table data to a list of lists (rows of cells)
    data = table.extract()

    # --- Generate Plain Text Table ---
    plain_text_rows = []
    # Determine the width for each column to align the text
    col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]
    
    for row in data:
        formatted_row = []
        for i, cell in enumerate(row):
            # Convert cell to string and justify it based on column width
            formatted_row.append(str(cell).ljust(col_widths[i]))
        plain_text_rows.append(" | ".join(formatted_row))
    
    plain_text_table = "\n".join(plain_text_rows)


    # --- Generate Markdown Table (GitHub Flavored Markdown) ---
    markdown_rows = []
    
    # 1. Header Row
    header = [str(cell) for cell in data[0]]
    markdown_rows.append("| " + " | ".join(header) + " |")

    # 2. Separator Row
    separator = ["---" for _ in header]
    markdown_rows.append("| " + " | ".join(separator) + " |")

    # 3. Data Rows
    for row in data[1:]:
        content = [str(cell).replace('\n', '<br>') for cell in row] # Use <br> for newlines in Markdown cells
        markdown_rows.append("| " + " | ".join(content) + " |")

    markdown_table = "\n".join(markdown_rows)

    return plain_text_table, markdown_table


def clean_latex_cell(text: str) -> str:
    """
    Cleans a string from a LaTeX table cell, removing common commands
    to extract the plain text content.
    """
    text = text.strip()
    
    # 1. Handle commands that wrap content in braces, e.g., \textbf{}, \text{}, \shortstack{}:
    text = re.sub(r'\\(?:text[a-z]+|bf|it|sc|sffamily|shortstack)\s*\{(.*?)\}', r'\1', text, flags=re.DOTALL)
    
    # 2. Handle \multirow{rows}{width}{content} and \multicolumn{cols}{align}{content}
    # Includes handling for the common typo \multIColumn
    def extract_last_brace_content(match):
        parts = re.findall(r'\{(.*?)\}', match.group(0), re.DOTALL)
        return parts[-1] if parts else ''

    # Combined both correct and misspelled versions of multicolumn/multirow
    text = re.sub(r'\\(?:multirow|multicolumn|multIColumn)\s*(\{.*?\})+\s*', extract_last_brace_content, text, flags=re.DOTALL)

    # 3. Remove math mode delimiters like $...$
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    
    # 3.5. Clean up nested tabular environment artifacts
    text = re.sub(r'\\begin\{tabular\}.*?\}', '', text) 
    text = re.sub(r'\\end\{tabular\}', '', text)     
    
    # 4. Remove commands with optional arguments in brackets/footnotes, e.g., [10]
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'\\(?:hline|cline|cmidrule|vspace|smallskip|cdot|label)\s*(\{.*?\})?', '', text, flags=re.DOTALL) 
    
    # 5. Remove any remaining simple commands that are not followed by braces or brackets.
    text = re.sub(r'\\([a-zA-Z]+)\s*', '', text)
    
    # 6. Clean up remaining leading/trailing whitespace and | characters
    text = text.replace('|', '').strip()

    return text.strip()

def process_latex_table(file_path: str) -> Tuple[str, int, int]:
    """
    Renders a LaTeX table from a .txt file to a PNG image. Includes column
    deletion and a parser that is more tolerant of missing row breaks or 
    a missing \end{tabular}.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        latex_content = f.read()

    # --- 1. Parse LaTeX table into a Python list of lists ---
    
    # Strategy 1: Find content between begin and end (most common, non-greedy)
    table_inner_content_match = re.search(
        r'\\begin{tabular}\{.*?\}\s*(.*?)\s*\\end{tabular}', 
        latex_content, 
        re.DOTALL | re.IGNORECASE
    )
    
    if not table_inner_content_match:
        # Strategy 2: Find content without a specific format string, up to \end{tabular}
        table_inner_content_match = re.search(
            r'\\begin{tabular}.*?\s*(.*?)\s*\\end{tabular}', 
            latex_content, 
            re.DOTALL
        )
    
    if not table_inner_content_match:
        # Strategy 3: Find content from \begin{tabular} to END OF STRING
        # This handles the case where \end{tabular} is completely missing.
        # We find the *last* instance of \begin{tabular} to ensure we get the right one.
        matches = list(re.finditer(r'\\begin{tabular}\{.*?\}|\\begin{tabular}', latex_content, re.DOTALL))
        if matches:
            last_start_match = matches[-1]
            # Capture everything from after the last \begin{tabular} to the end of the file
            table_inner_content = latex_content[last_start_match.end():].strip()
            # Clean up the format string part that might be left at the start
            table_inner_content = re.sub(r'\{.*?\}\s*', '', table_inner_content, 1, re.DOTALL)
            
            # Since we don't have \end{tabular}, we rely on the end of the file/string
            if not table_inner_content.strip():
                 raise ValueError("Could not find a valid tabular environment in the file.")
        else:
            raise ValueError("Could not find a valid tabular environment in the file.")
    else:
        table_inner_content = table_inner_content_match.group(1).strip()
    
    # -----------------------------------------------------------------
    # --- Row Parsing (Unchanged logic, but uses the more flexible content) ---
    # -----------------------------------------------------------------
    table_data = []
    all_lines = table_inner_content.split('\n')
    current_line_buffer = ""

    for line in all_lines:
        line = line.strip()
        
        if not line or line.startswith(r'%'):
            continue
            
        if re.fullmatch(r'\\(?:hline|cline|cmidrule|vspace|smallskip).*', line, re.DOTALL):
            continue
            
        current_line_buffer += (' ' if current_line_buffer and not current_line_buffer.endswith(' ') else '') + line.strip()
        
        # Look for row terminator or line command that implies row break
        is_complete_row = re.search(r'\\\\\s*$', current_line_buffer) or re.search(r'\\(?:hline|cline|cmidrule)', current_line_buffer)

        if is_complete_row:
            row_content = re.sub(r'\\\\\s*$', '', current_line_buffer).strip()
            row_content = re.sub(r'\\(?:hline|cline|cmidrule|vspace|smallskip)\s*(\{.*?\})?', '', row_content, flags=re.DOTALL).strip()
            
            if row_content:
                cells = [clean_latex_cell(cell) for cell in re.split(r'(?<!\\)&', row_content)]
                
                while cells and not cells[-1].strip():
                    cells.pop()
                    
                if cells:
                    table_data.append(cells)
                    
            current_line_buffer = ""
            
    # Process any remaining content (e.g., last line missing \\)
    if current_line_buffer.strip():
        row_content = re.sub(r'\\\\\s*$', '', current_line_buffer).strip()
        row_content = re.sub(r'\\(?:hline|cline|cmidrule|vspace|smallskip)\s*(\{.*?\})?', '', row_content, flags=re.DOTALL).strip()
        if row_content:
            cells = [clean_latex_cell(cell) for cell in re.split(r'(?<!\\)&', row_content)]
            while cells and not cells[-1].strip():
                cells.pop()
            if cells:
                table_data.append(cells)


    if not table_data:
        raise ValueError("The parsed table contains no data rows.")
    
    # --- 2. Determine canonical column count and pad/trim rows (Unchanged) ---
    all_row_lengths = [len(row) for row in table_data]
    col_counts = Counter(all_row_lengths)
    
    if not col_counts:
        raise ValueError("Could not determine canonical column count.")
        
    canonical_num_cols = col_counts.most_common(1)[0][0]
    
    for row in table_data:
        while len(row) < canonical_num_cols:
            row.append('')
        if len(row) > canonical_num_cols:
            row[:] = row[:canonical_num_cols]
    
    # --- 3. Column Deletion (Unchanged) ---
    cols_to_keep = []
    
    for col_idx in range(canonical_num_cols):
        is_empty = True
        for row in table_data:
            if row[col_idx].strip():
                is_empty = False
                break
        
        if not is_empty:
            cols_to_keep.append(col_idx)

    new_table_data = []
    for row in table_data:
        new_row = [row[idx] for idx in cols_to_keep]
        new_table_data.append(new_row)
        
    table_data = new_table_data

    # --- 4. Get final dimensions (Unchanged) ---
    num_rows = len(table_data)
    num_cols = len(table_data[0]) if table_data else 0

    if num_cols == 0:
          raise ValueError("All columns were empty after cleaning.")

    # --- 5. Render the table (Unchanged) ---
    output_path = f"{os.path.splitext(file_path)[0]}_rendered.png"
    
    font_prop = None
    cjk_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS', 'Heiti TC']
    try:
        font_path = fm.findfont(plt.rcParams['font.sans-serif'][0], fontext="ttf")
        font_prop = fm.FontProperties(fname=font_path)
    except Exception:
        for font in cjk_fonts:
            try:
                font_path = fm.findfont(font, fontext="ttf")
                font_prop = fm.FontProperties(fname=font_path)
                break
            except Exception:
                continue

    if not font_prop:
        font_prop = fm.FontProperties(family='sans-serif')

    fig, ax = plt.subplots(figsize=(num_cols * 1.5 + 2, num_rows * 0.4 + 2)) 
    ax.axis('tight')
    ax.axis('off')

    the_table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10) 
    the_table.scale(1.0, 1.2) 

    for cell in the_table.get_celld().values():
        cell.set_text_props(fontproperties=font_prop)

    title_text = f"渲染表格 (Rendered Table)\n列数 (Columns): {num_cols}, 行数 (Rows): {num_rows}"
    caption_match = re.search(r'\\caption\{(.*?)\}', latex_content, re.DOTALL)
    if caption_match:
        caption = clean_latex_cell(caption_match.group(1))
        title_text = f"{caption}\n{title_text}"
        
    plt.title(title_text, fontsize=12, pad=20, fontproperties=font_prop)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)

    return output_path, num_cols, num_rows


def process_markdown_table(file_path: str) -> Tuple[str, int, int]:
    """
    Process a single .txt file containing a Markdown table.
    Handles nested or merged headers by identifying the data part using separator and numeric detection.
    Renders the table as a PNG image in the same directory as the .txt file with '_rendered' suffix.
    Shows data dimensions in the rendered image.
    Returns the PNG path and the data dimensions (cols, rows).
    
    :param file_path: Path to the .txt file with Markdown table.
    :return: Tuple of (png_path, num_data_cols, num_data_rows)
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"{file_path} is not a valid file")

    directory = os.path.dirname(file_path)
    txt_file = os.path.basename(file_path)
    if not txt_file.endswith('.txt'):
        raise ValueError(f"{txt_file} is not a .txt file")

    try:
        # Parse the table from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Filter lines that look like table rows
        table_lines = [line for line in lines if line.startswith('|') and line.endswith('|')]
        if not table_lines:
            raise ValueError(f"No Markdown table found in {txt_file}")

        # Parse rows into cells
        cells = []
        for line in table_lines:
            # Split by |, remove outer empties, strip cells
            row = [cell.strip() for cell in line.split('|')[1:-1]]
            cells.append(row)

        original_cells = [row[:] for row in cells]  # copy for rendering

        # Find separator row index (contains ---)
        sep_idx = next((i for i, row in enumerate(cells) if any('---' in cell for cell in row)), None)

        num_cols = 0
        num_rows = 0

        if sep_idx is not None:
            # Data starts after sep
            data_start = sep_idx + 1
            data_rows = []
            for i in range(data_start, len(original_cells)):
                row = original_cells[i]
                if len(row) == 0:
                    continue
                # Assume first column is label, others values
                value_cells = row[1:]
                # Check if has numeric values (flexible regex for numbers, optional % or notes)
                num_count = sum(1 for cell in value_cells if re.match(r'^-?\d+(?:\.\d+)?(?:\%|\[.*\])?$', cell) is not None)
                if num_count >= len(value_cells) * 0.5:  # at least half are numeric
                    data_rows.append(row)
            
            if data_rows:
                num_rows = len(data_rows)
                num_cols = len(data_rows[0])
                # Verify consistency
                if any(len(row) != num_cols for row in data_rows):
                    raise ValueError(f"Inconsistent column count in data rows of {txt_file}")
        else:
            # Fallback: no sep, use most common row length for cols, largest consecutive block for rows
            lengths = [len(row) for row in cells]
            col_freq = Counter(lengths)
            if col_freq:
                candidate_cols = col_freq.most_common(1)[0][0]
                # Largest consecutive block
                max_block = 0
                curr = 0
                for l in lengths:
                    if l == candidate_cols:
                        curr += 1
                        max_block = max(max_block, curr)
                    else:
                        curr = 0
                num_cols = candidate_cols
                num_rows = max_block

        if num_cols == 0 or num_rows == 0:
            raise ValueError(f"No data rows found in {txt_file}")

        # For rendering: use original_cells, pad to rectangular
        all_rows = original_cells
        max_cols = max((len(row) for row in all_rows), default=0)
        for row in all_rows:
            row.extend([''] * (max_cols - len(row)))

        # Render with matplotlib
        fig, ax = plt.subplots(figsize=(max_cols * 1.5, len(all_rows) * 0.5))
        ax.axis('off')
        table = ax.table(cellText=all_rows, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Add dimensions text below the table
        plt.subplots_adjust(bottom=0.15)
        fig.text(0.5, 0.05, f'Data dimensions: {num_rows} rows × {num_cols} columns', 
                 ha='center', va='bottom', transform=fig.transFigure, fontsize=12)

        # Save PNG in the same directory as the .txt file
        base_name = os.path.splitext(txt_file)[0]
        png_path = os.path.join(directory, f"{base_name}_rendered.png")
        plt.savefig(png_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig)

        print(f"Rendered {txt_file} to {png_path}")

        return png_path, num_cols, num_rows

    except Exception as e:
        raise ValueError(f"Error processing {txt_file}: {e}")
    

if __name__ == '__main__':
    # table_image_folder = f'D:\\Codes\\2025\RAG\doc_analyzer\output\layout_detection\漳州统计年鉴2024_483-520\\table_images'
    table_image_folder = f'D:\\Codes\\2025\RAG\doc_analyzer\output\layout_detection\漳州统计年鉴2024_19-52\\table_images'
    # Process all .txt files in the folder using process_latex_table
    for filename in os.listdir(table_image_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(table_image_folder, filename)
            try:
                output_path, num_cols, num_rows = process_latex_table(file_path)
                print(f"Processed {filename}: {output_path} ({num_rows} rows, {num_cols} columns)")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")