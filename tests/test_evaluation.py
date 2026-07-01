import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moderation.evaluation import evaluate, precision_at_prevalence, wilson_ci  # noqa: E402


def test_evaluate_matriz_conocida():
    # 9 verdad: 5 positivos, 4 negativos. Prediccion con 1 FN y 1 FP.
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 1, 0, 0, 0]
    m = evaluate(y_true, y_pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (4, 1, 1, 3)
    assert m["precision"] == round(4 / 5, 4)
    assert m["recall"] == round(4 / 5, 4)


def test_wilson_ci_contiene_la_proporcion():
    lo, hi = wilson_ci(8, 10)
    assert lo < 0.8 < hi
    assert 0 <= lo <= hi <= 1


def test_precision_cae_a_baja_prevalencia():
    # Con recall y especificidad fijos, la precision baja al bajar la prevalencia.
    p_balanceada = precision_at_prevalence(0.95, 0.90, prevalence=0.5)
    p_real = precision_at_prevalence(0.95, 0.90, prevalence=0.056)
    assert p_real < p_balanceada
