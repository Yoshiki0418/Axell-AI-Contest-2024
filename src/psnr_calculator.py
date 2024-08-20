from torchvision import transforms
from torch import Tensor
import numpy as np
import cv2

to_image = transforms.ToPILImage()
def calc_psnr(image1: Tensor, image2: Tensor):
    image1 = cv2.cvtColor((np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor((np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return cv2.PSNR(image1, image2)