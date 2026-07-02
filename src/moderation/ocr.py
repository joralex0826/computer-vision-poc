"""Etapa de OCR local (RapidOCR / onnxruntime)."""
from .utils import clean_ocr


class OCRStage:
    """Extrae texto de una imagen usando RapidOCR en local."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()

    def __call__(self, image_path):
        """Devuelve el texto de la imagen como string limpio, o vacio si no hay texto."""
        res, _ = self.engine(str(image_path))
        # Se limpia aqui (una sola vez) para que el router, el LLM y la evidencia
        # reciban texto sin saltos de linea ni espacios repetidos.
        return clean_ocr(" ".join(r[1] for r in res)) if res else ""
