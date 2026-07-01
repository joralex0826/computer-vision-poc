import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moderation.utils import norm, parse_output  # noqa: E402


def test_parse_output_valido():
    r = parse_output('{"has_infraction": true, "reason": "promo"}')
    assert r["has_infraction"] is True
    assert r["reason"] == "promo"


def test_parse_output_con_texto_extra():
    r = parse_output('Claro, aqui: {"has_infraction": false} listo')
    assert r["has_infraction"] is False


def test_parse_output_invalido():
    assert parse_output("no hay json aqui") is None


def test_norm_colapsa_espacios_y_saltos():
    assert norm("mercado\n  Lider") == "mercado lider"
