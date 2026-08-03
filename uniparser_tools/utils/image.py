"""
- dump String: Array > Image
- load String: Image >>> Array
- dump & load: Image > Array
"""

import base64
import os
import struct
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


default_font_path = os.path.join(os.path.dirname(__file__), "DejaVuSerif.ttf")


def _prepare_image_for_jpeg(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def dump_image_base64_str(image: Image.Image, quality: int = 85) -> str:
    img_byte_arr = BytesIO()
    _prepare_image_for_jpeg(image).save(img_byte_arr, format="JPEG", quality=quality)
    return base64.b64encode(img_byte_arr.getvalue()).decode("ascii")


def load_image_base64_str(base64_string: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_string)))


def dump_array_base64_str(image: np.ndarray, quality: int = 85) -> str:
    img_byte_arr = cv2.imencode(".jpeg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    return base64.b64encode(img_byte_arr).decode("ascii")


def load_array_base64_str(base64_string: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(base64_string), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def get_image_size_from_base64_str(b64: str) -> Tuple[int, int]:
    head = base64.b64decode(b64[:128])  # 只解前 128 个字符足够
    # --- PNG ---
    if head.startswith(b"\x89PNG"):
        w, h = struct.unpack(">II", head[16:24])
        return w, h
    # --- JPEG  SOI + APP0/APP1 ---
    if head.startswith(b"\xff\xd8"):
        # 简单策略：直接搜 0xFFC0 / 0xFFC2 帧
        idx = 0
        while idx < len(head) - 9:
            if head[idx] == 0xFF and head[idx + 1] == 0xC0 or head[idx + 1] == 0xC2:
                h, w = struct.unpack(">HH", head[idx + 5 : idx + 9])
                return w, h
            idx += 1
    # --- WebP ---
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        if head[12:16] == b"VP8 ":
            w, h = struct.unpack("<HH", head[26:30])
            return w & 0x3FFF, h & 0x3FFF
        if head[12:16] == b"VP8X":
            w = 1 + int.from_bytes(head[24:27], "little")
            h = 1 + int.from_bytes(head[27:30], "little")
            return w, h
    # 兜底：全解码一次
    try:
        return load_image_base64_str(b64).size
    except Exception:
        return None
