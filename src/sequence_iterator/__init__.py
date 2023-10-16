import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

import numpy as np
import cv2
import os
import shutil

class ImageSequenceIterator:
    
    def __init__(
            self,
            input_sequence_prefix="/tmp/input",
            output_sequence_prefix="/tmp/output",
            image_extension="png",
            logging_level=logging.WARNING):
        self.input_sequence_prefix = input_sequence_prefix
        self.output_sequence_prefix = output_sequence_prefix
        self.image_extension = image_extension
        logger.setLevel(logging_level)

    def process(
            self,
            image,
            image_filename,
            info_filename):
        raise NotImplementedError("Subclasses must implement this method.")

    def process_sequence(self):
        list_of_imagenames = np.array(
            [img for img in os.listdir(self.input_sequence_prefix)
             if self.image_extension in img])
        for image_name in list_of_imagenames:
            input_img_name = f"{self.input_sequence_prefix}/{image_name}"
            image = cv2.imread(input_img_name, cv2.IMREAD_UNCHANGED)
            logger.info(f"Reading {input_img_name} {image.dtype} min={np.min(image)} max={np.max(image)} avg={np.average(image)}")
            input_img_name_no_extension = input_img_name.split(".")[0]
            input_info_file = f"{input_img_name_no_extension + '_info.txt'}"
            processed_image, info = self.process(image, input_img_name, input_info_file)
            output_img_name = f"{self.output_sequence_prefix}/{image_name}"
            logger.info(f"Saving {output_img_name} {processed_image.dtype} min={np.min(processed_image)} max={np.max(processed_image)} avg={np.average(processed_image)} info={info}")
            cv2.imwrite(output_img_name, processed_image)
            output_img_name_no_extension = output_img_name.split(".")[0]
            output_info_file = f"{output_img_name_no_extension + '_info.txt'}"
            logger.debug(f"{input_info_file} existence: {os.path.exists(input_info_file)}")
            if os.path.exists(input_info_file):
                shutil.copy(input_info_file, output_info_file)
            with open(output_info_file, 'a') as f:
                f.write(info + '\n')

