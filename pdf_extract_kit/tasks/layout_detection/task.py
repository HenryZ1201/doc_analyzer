from pdf_extract_kit.registry.registry import TASK_REGISTRY
from pdf_extract_kit.tasks.base_task import BaseTask


@TASK_REGISTRY.register("layout_detection")
class LayoutDetectionTask(BaseTask):
    def __init__(self, model):
        super().__init__(model)
        self.pdf_images = None
        self.pdf_images_info = None  # to store width, height, dpi info for each image

    def predict_images(self, input_data, result_path):
        """
        Predict layouts in images.

        Args:
            input_data (str): Path to a single image file or a directory containing image files.
            result_path (str): Path to save the prediction results.

        Returns:
            list: List of prediction results.
        """
        images = self.load_images(input_data)
        # Perform detection
        return self.model.predict(images, result_path)

    def predict_pdfs(self, input_data, result_path):
        """
        Predict layouts in PDF files.

        Args:
            input_data (str): Path to a single PDF file or a directory containing PDF files.
            result_path (str): Path to save the prediction results.

        Returns:
            list: List of prediction results.
        """
        pdf_images, pdf_images_info = self.load_pdf_images(input_data)   # image_keys are image ids formed by pdf name and page number (starting from 1)
        self.pdf_images = pdf_images
        self.pdf_images_info = pdf_images_info
        # Perform detection
        return self.model.predict(list(pdf_images.values()), result_path, list(pdf_images.keys()))
    
    def get_pdf_images_info(self):
        """
        Get the PDF images info including width, height, and dpi for each image.

        Returns:
            dict: Dictionary with image IDs as keys and their corresponding info (width, height, dpi) as values.
        """
        return self.pdf_images_info