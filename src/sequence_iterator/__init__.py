import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

import numpy as np
import cv2
import os

class ImageSequenceIterator:
    
    def __init__(self,
                 input_sequence_prefix="/tmp/input",
                 output_sequence_prefix="/tmp/output",
                 image_extension="png"):
        self.input_sequence_prefix = input_sequence_prefix
        self.output_sequence_prefix = output_sequence_prefix
        self.image_extension = image_extension

    def process(self,
                image):
        raise NotImplementedError("Subclasses must implement this method.")

    def process_sequence(self):
        list_of_imagenames = np.array(
            [img for img in os.listdir(self.input_sequence_prefix)
             if self.image_extension in img])
        for image_name in list_of_imagenames:
            img_name = f"{self.input_sequence_prefix}/{image_name}"
            image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            logger.debug(f"Reading {img_name} {image.dtype} {np.min(image)} {np.max(image)} {np.average(image)}")
            processed_image = self.process(image)
            img_name = f"{self.output_sequence_prefix}/{image_name}"
            logger.debug(f"Saving {img_name} {processed_image.dtype} {np.min(processed_image)} {np.max(processed_image)} {np.average(processed_image)}")
            cv2.imwrite(img_name, processed_image)
