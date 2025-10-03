# import fitz
# from PIL import Image

# def load_pdf_page(page, dpi):
#     """
#     功能：将传入的 PDF 页面对象（page）渲染为指定 DPI 的图片。
#     实现细节：
#         使用 fitz.Matrix(dpi/72, dpi/72) 设置渲染分辨率（DPI）。
#         用 page.get_pixmap() 渲染页面为像素图（Pixmap）。
#         将像素图转换为 PIL 的 Image 对象。
#         如果图片宽或高大于 3000 像素，则用较低分辨率（1,1）重新渲染，避免内存占用过大。
#         返回值：PIL Image 对象。
#     """
#     pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
#     image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     effective_dpi = dpi
#     if pix.width > 3000 or pix.height > 3000:
#         pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
#         image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         effective_dpi = 72  # Fallback to 72 DPI
#     return image, pix, effective_dpi, pix.width, pix.height

# def load_pdf(pdf_path, dpi=144):
#     """
#     功能：将整个 PDF 文件的每一页都转换为图片。
#     实现细节：
#         打开 PDF 文件，遍历每一页。
#         对每一页调用 load_pdf_page，得到图片。
#         将所有页面图片存入列表并返回。
#         返回值：包含所有页面图片（PIL Image）的列表。
#     """
#     images = []
#     doc = fitz.open(pdf_path)
#     for i in range(len(doc)):
#         page = doc[i]
#         # get pdf page properties
#         pdf_width, pdf_height = page.mediabox.width, page.mediabox.height
#         rotation = page.rotation
#         rotation = page.rotation
#         image, pix, effective_dpi, width, height = load_pdf_page(page, dpi)
#         images.append((image, width, height, effective_dpi, pdf_width, pdf_height, rotation))
#     doc.close()
#     return images

# import fitz
# from PIL import Image

# def load_pdf_page(page, dpi):
#     """
#     功能：将传入的 PDF 页面对象（page）渲染为指定 DPI 的图片。
#     实现细节：
#         使用 fitz.Matrix(dpi/72, dpi/72) 设置渲染分辨率（DPI）。
#         用 page.get_pixmap() 渲染页面为像素图（Pixmap）。
#         将像素图转换为 PIL 的 Image 对象。
#         如果图片宽或高大于 3000 像素，则用较低分辨率（1,1）重新渲染，避免内存占用过大。
#         返回值：PIL Image 对象。
#     """
#     pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
#     image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     effective_dpi = dpi
#     if pix.width > 3000 or pix.height > 3000:
#         pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
#         image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         effective_dpi = 72  # Fallback to 72 DPI
#     return image, pix, effective_dpi, pix.width, pix.height

# def load_pdf(pdf_path, dpi=144):
#     """
#     功能：将整个 PDF 文件的每一页都转换为图片。
#     实现细节：
#         打开 PDF 文件，遍历每一页。
#         对每一页调用 load_pdf_page，得到图片。
#         将所有页面图片存入列表并返回。
#         返回值：包含所有页面图片（PIL Image）的列表。
#     """
#     images = []
#     doc = fitz.open(pdf_path)
#     for i in range(len(doc)):
#         page = doc[i]
#         # get pdf page properties
#         pdf_width, pdf_height = page.mediabox.width, page.mediabox.height
#         rotation = page.rotation
#         rotation = page.rotation
#         image, pix, effective_dpi, width, height = load_pdf_page(page, dpi)
#         images.append((image, width, height, effective_dpi, pdf_width, pdf_height, rotation))
#     doc.close()
#     return images

import fitz
from PIL import Image
import os

def load_pdf_page(page, dpi):
    """
    功能：将传入的 PDF 页面对象（page）渲染为指定 DPI 的图片。
    实现细节：
        使用 fitz.Matrix(dpi/72, dpi/72) 设置渲染分辨率（DPI）。
        用 page.get_pixmap() 渲染页面为像素图（Pixmap）。
        将像素图转换为 PIL 的 Image 对象。
        如果图片宽或高大于 3000 像素，则用较低分辨率（1,1）重新渲染，避免内存占用过大。
        返回值：PIL Image 对象。
    """
    scale_factor = dpi / 72.0
    # print(f"DEBUG_LOAD: Attempting render with DPI={dpi} (Scale={scale_factor:.4f})")
    
    pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    effective_dpi = dpi
    
    if pix.width > 3000 or pix.height > 3000:
        # print(f"DEBUG_LOAD: --- Fallback Triggered: Initial dimensions {pix.width}x{pix.height}px exceed 3000px. ---")
        # Fallback to 72 DPI (scale 1.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        effective_dpi = 72  # Fallback to 72 DPI
        # print(f"DEBUG_LOAD: Fallback Render Dimensions: {pix.width} x {pix.height} pixels")

    # print(f"DEBUG_LOAD: Effective DPI: {effective_dpi}, Image Dimensions: {pix.width} x {pix.height} pixels")
    return image, pix, effective_dpi, pix.width, pix.height

def load_pdf(pdf_path, dpi=144):
    """
    功能：将整个 PDF 文件的每一页都转换为图片。
    实现细节：
        打开 PDF 文件，遍历每一页。
        对每一页调用 load_pdf_page，得到图片。
        将所有页面图片存入列表并返回。
        返回值：包含所有页面图片（PIL Image）的列表。
    """
    images = []
    doc = fitz.open(pdf_path)
    # print(f"DEBUG_LOAD: Loading PDF: {pdf_path} with target DPI: {dpi}")
    
    for i in range(len(doc)):
        page = doc[i]
        # print(f"\nDEBUG_LOAD: --- Processing Page {i} ---")

        # get pdf page properties
        pdf_width, pdf_height = page.mediabox.width, page.mediabox.height
        rotation = page.rotation
        cropbox = page.cropbox
        page_rect = page.rect # --- 新增: 获取页面的有效矩形区域 (fitz.Rect) ---
        
        # print(f"DEBUG_LOAD: PDF Page Dimensions (MediaBox): {pdf_width:.2f} x {pdf_height:.2f} points")
        # print(f"DEBUG_LOAD: PDF Page Content Rect (Rect): {page.rect}")
        # print(f"DEBUG_LOAD: PDF Page CropBox: {cropbox}")
        # print(f"DEBUG_LOAD: PDF Page Rotation: {rotation}°")
        
        image, pix, effective_dpi, width, height = load_pdf_page(page, dpi)
        # Note: The original code used pdf_width/height from mediabox
        # --- 新增 page_rect 到返回元组中 ---
        images.append((image, width, height, effective_dpi, pdf_width, pdf_height, rotation, page_rect))
        
    doc.close()
    # print("DEBUG_LOAD: PDF document closed.")
    return images

