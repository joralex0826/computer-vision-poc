"""Tests de OCRStage."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moderation.ocr import OCRStage  # noqa: E402


def _make_ocr(engine_result):
    """OCRStage con el motor mockeado que devuelve engine_result."""
    # RapidOCR se importa de forma lazy dentro de __init__, hay que parchear
    # el modulo donde se importa (rapidocr_onnxruntime), no el de ocr.
    with patch.dict("sys.modules", {"rapidocr_onnxruntime": MagicMock()}):
        stage = OCRStage()
    stage.engine = MagicMock(return_value=(engine_result, None))
    return stage


def test_ocr_devuelve_texto_limpio(tmp_path):
    """El texto de multiples bloques se une y se limpia."""
    img = tmp_path / "img.jpg"
    img.write_bytes(b"fake")
    stage = _make_ocr([[None, "ENVIO  GRATIS"], [None, "hoy"]])
    result = stage(img)
    assert result == "ENVIO GRATIS hoy"


def test_ocr_devuelve_vacio_si_sin_texto(tmp_path):
    """Imagen sin texto detectado -> string vacio."""
    img = tmp_path / "img.jpg"
    img.write_bytes(b"fake")
    stage = _make_ocr(None)
    assert stage(img) == ""
