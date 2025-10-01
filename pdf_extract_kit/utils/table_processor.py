import os
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Tuple, List
from collections import Counter


# Set Chinese font globally for Matplotlib to display Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False # Correctly display minus signs

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















from collections import Counter
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