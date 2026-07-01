"""Etapa de OCR local (RapidOCR / onnxruntime). Sin dependencias de terceros."""
from .utils import clean_ocr


class OCRStage:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()

    def __call__(self, image_path):
        res, _ = self.engine(str(image_path))
        # Se limpia aqui (una sola vez) para que el router, el LLM y la evidencia
        # reciban texto sin saltos de linea ni espacios repetidos.
        return clean_ocr(" ".join(r[1] for r in res)) if res else ""
