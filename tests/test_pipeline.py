"""Tests del pipeline: descarga de imagenes y logica de la cascada."""
import sys
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moderation.pipeline import ModerationPipeline  # noqa: E402


# --- helpers ------------------------------------------------------------------

def _make_pipe(ocr_text="", suspicious=False, vlm_result=None):
    """Construye un pipeline con todas las etapas mockeadas."""
    with patch("moderation.pipeline.OCRStage"), \
         patch("moderation.pipeline.RouterStage"), \
         patch("moderation.pipeline.Moderator"):
        pipe = ModerationPipeline()

    pipe.ocr = MagicMock(return_value=ocr_text)
    pipe.router = MagicMock()
    pipe.router.is_suspicious = MagicMock(return_value=suspicious)
    pipe.vlm = MagicMock()
    pipe.vlm.moderate = MagicMock(return_value=vlm_result or {
        "has_infraction": True, "evidence": "franja 50% OFF", "reason": "promo", "confidence": "high"
    })
    return pipe


# --- _resolve_image -----------------------------------------------------------

def test_resolve_image_path_local_no_descarga(tmp_path):
    """Si se pasa image_path no debe tocar la red."""
    img = tmp_path / "foto.jpg"
    img.write_bytes(b"fake")
    with patch("moderation.pipeline.OCRStage"), \
         patch("moderation.pipeline.RouterStage"), \
         patch("moderation.pipeline.Moderator"):
        pipe = ModerationPipeline()
    result = pipe._resolve_image(None, str(img))
    assert result == img


def test_resolve_image_usa_cache(tmp_path):
    """Si la imagen ya esta en cache no vuelve a descargarla."""
    with patch("moderation.pipeline.OCRStage"), \
         patch("moderation.pipeline.RouterStage"), \
         patch("moderation.pipeline.Moderator"), \
         patch("moderation.pipeline.CACHE_DIR", tmp_path):
        pipe = ModerationPipeline()
        url = "https://ejemplo.com/img.jpg"
        dest = tmp_path / (hashlib.md5(url.encode()).hexdigest() + ".jpg")
        dest.write_bytes(b"cached")
        with patch("requests.get") as mock_get:
            pipe._resolve_image(url, None)
            mock_get.assert_not_called()


def test_resolve_image_falla_si_status_error(tmp_path):
    """Un 404 debe lanzar excepcion y no dejar archivo temporal."""
    import requests
    with patch("moderation.pipeline.OCRStage"), \
         patch("moderation.pipeline.RouterStage"), \
         patch("moderation.pipeline.Moderator"), \
         patch("moderation.pipeline.CACHE_DIR", tmp_path):
        pipe = ModerationPipeline()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                pipe._resolve_image("https://ejemplo.com/404.jpg", None)
        # el .tmp no debe quedar huerfano
        assert not any(tmp_path.glob("*.tmp"))


# --- moderate: cortes de la cascada ------------------------------------------

def test_moderate_corta_en_ocr_si_sin_texto(tmp_path):
    """Sin texto el pipeline devuelve limpio desde la etapa OCR."""
    img = tmp_path / "clean.jpg"
    img.write_bytes(b"fake")
    pipe = _make_pipe(ocr_text="")
    result = pipe.moderate(image_path=str(img))
    assert result["has_infraction"] is False
    assert result["stage"] == "ocr"
    pipe.router.is_suspicious.assert_not_called()
    pipe.vlm.moderate.assert_not_called()


def test_moderate_corta_en_router_si_texto_neutro(tmp_path):
    """Texto presente pero neutro: el router lo descarta y el VLM no se invoca."""
    img = tmp_path / "product.jpg"
    img.write_bytes(b"fake")
    pipe = _make_pipe(ocr_text="Camiseta talla M color azul", suspicious=False)
    result = pipe.moderate(image_path=str(img))
    assert result["has_infraction"] is False
    assert result["stage"] == "router"
    pipe.vlm.moderate.assert_not_called()


def test_moderate_llega_al_vlm_si_sospechoso(tmp_path):
    """Texto sospechoso: el VLM se invoca y su resultado se devuelve."""
    img = tmp_path / "promo.jpg"
    img.write_bytes(b"fake")
    vlm_resp = {"has_infraction": True, "evidence": "50% OFF sobrepuesto",
                "reason": "promo", "confidence": "high"}
    pipe = _make_pipe(ocr_text="50% OFF hoy", suspicious=True, vlm_result=vlm_resp)
    result = pipe.moderate(image_path=str(img))
    assert result["has_infraction"] is True
    assert result["stage"] == "vlm"
    assert result["reason"] == "promo"
    pipe.vlm.moderate.assert_called_once()


def test_moderate_incluye_latencias(tmp_path):
    """La respuesta siempre debe traer latency_ms y stage_ms."""
    img = tmp_path / "img.jpg"
    img.write_bytes(b"fake")
    pipe = _make_pipe(ocr_text="")
    result = pipe.moderate(image_path=str(img))
    assert "latency_ms" in result
    assert "stage_ms" in result
    assert result["latency_ms"] >= 0
