import fitz
from PIL import Image


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
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if pix.width > 3000 or pix.height > 3000:
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image

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
    for i in range(len(doc)):
        page = doc[i]
        image = load_pdf_page(page, dpi)
        images.append(image)
    return images