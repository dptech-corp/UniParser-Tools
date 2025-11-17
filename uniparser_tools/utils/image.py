"""
- dump String: Array > Image
- load String: Image >>> Array
- dump & load: Image > Array
"""

import base64
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


default_font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")


def dump_image_base64_str(image: Image.Image, quality: int = 85) -> str:
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG", quality=quality)
    return base64.b64encode(img_byte_arr.getvalue()).decode("ascii")


def load_image_base64_str(base64_string: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_string)))


def dump_array_base64_str(image: np.ndarray, quality: int = 85) -> str:
    img_byte_arr = cv2.imencode(".jpeg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    return base64.b64encode(img_byte_arr).decode("ascii")


def load_array_base64_str(base64_string: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(base64_string), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
