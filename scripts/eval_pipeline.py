import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from moderation import MODEL  # noqa: E402
from moderation import ModerationPipeline  # noqa: E402
from moderation.evaluation import evaluate  # noqa: E402

GOLDEN = ROOT / "data" / "golden" / "golden_set.parquet"
MANIFEST = ROOT / "data" / "golden" / "images_manifest.parquet"
OUT = ROOT / "data" / "golden" / "pipeline_preds.parquet"
ENGINE_MODEL = os.environ.get("VLM_MODEL", MODEL)

SUBSET = sys.argv[1] if len(sys.argv) > 1 else "all"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None


def main():
    g = pd.read_parquet(GOLDEN)
    paths = pd.read_parquet(MANIFEST).set_index("picture_url")["local_path"].to_dict()
    if SUBSET == "reviewed":
        g = g[g["human_label"].notna()].copy()
    if LIMIT:
        g = g.sample(min(LIMIT, len(g)), random_state=42)
    g = g.reset_index(drop=True)
    print(f"subset={SUBSET} | imagenes={len(g)} | VLM={ENGINE_MODEL}", flush=True)

    # reanudacion: reusar resultados ya calculados si OUT existe
    done = {}
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        for r in prev.itertuples():
            done[r.picture_url] = (bool(r.pipeline_pred), r.stage, float(r.latency_ms))

    cols = ["picture_url", "pipeline_pred", "stage", "final_label", "legacy_label", "latency_ms"]

    def save():
        out = g.copy()
        out["pipeline_pred"] = out["picture_url"].map(lambda u: done.get(u, (None,))[0])
        out["stage"] = out["picture_url"].map(lambda u: done.get(u, (None, None))[1])
        out["latency_ms"] = out["picture_url"].map(lambda u: done.get(u, (None, None, None))[2])
        out[cols].to_parquet(OUT, index=False)

    pending = [u for u in g["picture_url"] if u not in done]
    print(f"pendientes: {len(pending)} / {len(g)}", flush=True)
    pipe = ModerationPipeline(vlm_model=ENGINE_MODEL) if pending else None

    t0, n_run = time.time(), 0
    for i, u in enumerate(g["picture_url"], 1):
        if u not in done:
            out = pipe.moderate(image_path=ROOT / paths[u])
            done[u] = (out["has_infraction"], out["stage"], out["latency_ms"])
            n_run += 1
            if n_run % 25 == 0:
                save()
                print(f"{i}/{len(g)}  {(time.time() - t0) / n_run:.1f}s/img (nuevas)", flush=True)
    save()

    g["pipeline_pred"] = g["picture_url"].map(lambda u: done[u][0])
    g["stage"] = g["picture_url"].map(lambda u: done[u][1])
    g["latency_ms"] = g["picture_url"].map(lambda u: done[u][2])
    lats = g["latency_ms"].tolist()
    stages = g["stage"].tolist()

    n = len(g)
    sc = Counter(stages)
    print("\n== etapa donde se decidio (ahorro de VLM) ==", flush=True)
    for st in ["ocr", "router", "vlm"]:
        print(f"  {st:7s}: {sc.get(st,0):4d}  ({sc.get(st,0)/n*100:.1f}%)", flush=True)
    print(f"  --> VLM solo se uso en {sc.get('vlm',0)/n*100:.1f}% de las imagenes", flush=True)

    lat = np.array(lats)
    print("\n== LATENCIA por imagen (ms) [hardware: este equipo] ==", flush=True)
    print(f"  p50={np.percentile(lat,50):.0f}  p95={np.percentile(lat,95):.0f}  "
          f"p99={np.percentile(lat,99):.0f}  max={lat.max():.0f}", flush=True)
    vlm_lat = g.loc[g["stage"] == "vlm", "latency_ms"]
    cheap_lat = g.loc[g["stage"] != "vlm", "latency_ms"]
    if len(cheap_lat):
        print(f"  etapas baratas (ocr/router): p99={np.percentile(cheap_lat,99):.0f}ms", flush=True)
    if len(vlm_lat):
        print(f"  con VLM: p99={np.percentile(vlm_lat,99):.0f}ms", flush=True)

    print("\n== PIPELINE COMPLETO vs verdad ==", flush=True)
    print(evaluate(g["final_label"].astype(bool), g["pipeline_pred"].astype(bool)), flush=True)
    print("\n== LEGACY (baseline) ==", flush=True)
    print(evaluate(g["final_label"].astype(bool), g["legacy_label"].astype(bool)), flush=True)


if __name__ == "__main__":
    main()
