"""Tests de RouterStage."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moderation.router import RouterStage  # noqa: E402


def _make_router():
    """RouterStage con mlx_lm mockeado (la libreria no esta disponible en CI)."""
    mock_mlx = MagicMock()
    mock_mlx.load.return_value = (MagicMock(), MagicMock())
    with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
        router = RouterStage()
    router.tok.apply_chat_template = MagicMock(return_value="prompt")
    return router, mock_mlx


def test_keyword_match_sin_llamar_al_llm():
    """Si hay un keyword directo no debe consultarse el LLM."""
    router, mock_mlx = _make_router()
    mock_mlx.generate = MagicMock()
    with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
        result = router.is_suspicious("ENVIO GRATIS disponible")
    mock_mlx.generate.assert_not_called()
    assert result is True


def test_llm_revisar_devuelve_true():
    """Si el LLM responde REVISAR, is_suspicious debe ser True."""
    router, mock_mlx = _make_router()
    mock_mlx.generate.return_value = "REVISAR"
    with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
        result = router.is_suspicious("mitad de precio solo hoy")
    assert result is True


def test_llm_limpio_devuelve_false():
    """Texto neutro sin keywords: el LLM dice LIMPIO y debe devolver False."""
    router, mock_mlx = _make_router()
    mock_mlx.generate.return_value = "LIMPIO"
    with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
        result = router.is_suspicious("Auriculares inalambricos color negro")
    assert result is False
