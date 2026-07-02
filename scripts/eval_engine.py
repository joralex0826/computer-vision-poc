import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from moderation import MODEL, Moderator  # noqa: E402

#   VLM_MODEL=mlx-community/Qwen2.5-VL-7B-Instruct-4bit python scripts/eval_engine.py all 150
ENGINE_MODEL = os.environ.get("VLM_MODEL", MODEL)

GOLDEN = ROOT / "data" / "golden" / "golden_set.parquet"
MANIFEST = ROOT / "data" / "golden" / "images_manifest.parquet"
OUT = ROOT / "data" / "golden" / "engine_preds.parquet"

# Uso: eval_engine.py [subset] [n]
#   subset: "reviewed" (solo etiqueta humana independiente) o "all" (700)
#   n: opcional, tamano de muestra aleatoria para una prueba rapida (ej. 50)
SUBSET = sys.argv[1] if len(sys.argv) > 1 else "reviewed"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
PREVALENCE = 27256 / 489639


def metrics(yt, yp):
    yt = np.asarray(yt, dtype=bool)
    yp = np.asarray(yp, dtype=bool)
    tp = int((yt & yp).sum()); fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum()); tn = int((~yt & ~yp).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    fpr = 1 - spec
    prec_real = (PREVALENCE * rec) / (PREVALENCE * rec + (1 - PREVALENCE) * fpr) if (PREVALENCE * rec + (1 - PREVALENCE) * fpr) else float("nan")
    return dict(n=len(yt), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(prec, 3), recall=round(rec, 3),
                precision_real_prev=round(prec_real, 3))


def main():
    g = pd.read_parquet(GOLDEN)
    paths = pd.read_parquet(MANIFEST).set_index("picture_url")["local_path"].to_dict()
    if SUBSET == "reviewed":
        g = g[g["human_label"].notna()].copy()
    if LIMIT:
        g = g.sample(min(LIMIT, len(g)), random_state=42)
    g = g.reset_index(drop=True)
    print(f"subset={SUBSET} | imagenes={len(g)} | modelo={ENGINE_MODEL}", flush=True)

    mod = Moderator(model=ENGINE_MODEL)
    preds = []
    t0 = time.time()
    for i, u in enumerate(g["picture_url"], 1):
        r = mod.moderate(ROOT / paths[u])
        preds.append(r["has_infraction"])
        if i % 20 == 0:
            print(f"{i}/{len(g)}  {(time.time() - t0) / i:.1f}s/img", flush=True)

    g["engine_pred"] = preds
    g[["picture_url", "engine_pred", "final_label", "legacy_label"]].to_parquet(OUT, index=False)

    print("\n== MOTOR (few-shot) vs verdad ==", flush=True)
    print(metrics(g["final_label"].astype(bool), g["engine_pred"].astype(bool)), flush=True)
    print("\n== LEGACY vs verdad (baseline) ==", flush=True)
    print(metrics(g["final_label"].astype(bool), g["legacy_label"].astype(bool)), flush=True)


if __name__ == "__main__":
    main()
