import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from moderation import MODEL, VERIFY_PROMPT, Moderator  # noqa: E402

PREDS = ROOT / "data" / "golden" / "engine_preds.parquet"
MANIFEST = ROOT / "data" / "golden" / "images_manifest.parquet"
OUT = ROOT / "data" / "golden" / "cascade_preds.parquet"
ENGINE_MODEL = os.environ.get("VLM_MODEL", MODEL)
PREVALENCE = 27256 / 489639


def metrics(yt, yp):
    yt = np.asarray(yt, dtype=bool); yp = np.asarray(yp, dtype=bool)
    tp = int((yt & yp).sum()); fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum()); tn = int((~yt & ~yp).sum())
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    fpr = 1 - spec
    Preal = (PREVALENCE * R) / (PREVALENCE * R + (1 - PREVALENCE) * fpr) if (PREVALENCE * R + (1 - PREVALENCE) * fpr) else float("nan")
    F1 = 2 * P * R / (P + R) if (P and R and not np.isnan(P) and not np.isnan(R)) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(P, 3),
                recall=round(R, 3), f1=round(F1, 3), precision_real_prev=round(Preal, 3))


def main():
    g = pd.read_parquet(PREDS)
    paths = pd.read_parquet(MANIFEST).set_index("picture_url")["local_path"].to_dict()
    flagged = g[g["engine_pred"].astype(bool)]
    print(f"etapa 1 marco {len(flagged)} positivos de {len(g)} | modelo={ENGINE_MODEL}", flush=True)

    mod = Moderator(model=ENGINE_MODEL, prompt=VERIFY_PROMPT)
    confirmed = {}
    t0 = time.time()
    for i, u in enumerate(flagged["picture_url"], 1):
        r = mod.moderate(ROOT / paths[u])
        confirmed[u] = r["has_infraction"]
        if i % 20 == 0:
            print(f"{i}/{len(flagged)}  {(time.time() - t0) / i:.1f}s/img", flush=True)

    # cascada: positivo solo si etapa 1 marco Y etapa 2 confirma
    g["cascade_pred"] = g.apply(
        lambda r: bool(r["engine_pred"]) and confirmed.get(r["picture_url"], True), axis=1)
    g.to_parquet(OUT, index=False)

    print("\n== ETAPA 1 sola (7B recall) ==", flush=True)
    print(metrics(g["final_label"].astype(bool), g["engine_pred"].astype(bool)), flush=True)
    print("\n== CASCADA (etapa 1 + verificador) ==", flush=True)
    print(metrics(g["final_label"].astype(bool), g["cascade_pred"].astype(bool)), flush=True)
    print("\n== LEGACY (baseline) ==", flush=True)
    print(metrics(g["final_label"].astype(bool), g["legacy_label"].astype(bool)), flush=True)


if __name__ == "__main__":
    main()
