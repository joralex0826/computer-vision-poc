import sys
import time
from pathlib import Path

import pandas as pd
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from moderation import MODEL, PROMPT, parse_output  # noqa: E402

CANDIDATES = ROOT / "data" / "golden" / "golden_candidates.parquet"
ANNOTATED = ROOT / "data" / "golden" / "golden_annotated.parquet"
MANIFEST = ROOT / "data" / "golden" / "images_manifest.parquet"

SAVE_EVERY = 20


def main():
    g = pd.read_parquet(ANNOTATED if ANNOTATED.exists() else CANDIDATES)
    man = pd.read_parquet(MANIFEST)
    paths = man.set_index("picture_url")["local_path"].to_dict()

    for c in ["vlm_has_infraction", "vlm_reason", "vlm_evidence", "vlm_confidence"]:
        if c not in g.columns:
            g[c] = pd.NA

    pending = g[g["vlm_has_infraction"].isna()].index
    print(f"pendientes: {len(pending)} / {len(g)}", flush=True)
    if len(pending) == 0:
        print("nada que anotar, todo listo.", flush=True)
        return

    model, processor = load(MODEL)
    config = load_config(MODEL)

    t0 = time.time()
    for n, i in enumerate(pending, 1):
        rel = paths.get(g.at[i, "picture_url"])
        if not rel:
            continue
        path = str(ROOT / rel)
        prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
        out = generate(model, processor, prompt, image=[path], max_tokens=200, verbose=False)
        text = out.text if hasattr(out, "text") else str(out)
        parsed = parse_output(text)
        if parsed:
            g.at[i, "vlm_has_infraction"] = bool(parsed.get("has_infraction"))
            g.at[i, "vlm_reason"] = str(parsed.get("reason"))
            g.at[i, "vlm_evidence"] = str(parsed.get("evidence"))
            g.at[i, "vlm_confidence"] = str(parsed.get("confidence"))
        else:
            g.at[i, "vlm_evidence"] = "PARSE_ERROR: " + text[:200]

        if n % SAVE_EVERY == 0:
            g.to_parquet(ANNOTATED, index=False)
            rate = (time.time() - t0) / n
            eta = rate * (len(pending) - n) / 60
            print(f"{n}/{len(pending)}  {rate:.1f}s/img  ETA {eta:.0f}min", flush=True)

    g.to_parquet(ANNOTATED, index=False)
    print("listo. resumen vlm_has_infraction:", flush=True)
    print(g["vlm_has_infraction"].value_counts(dropna=False).to_dict(), flush=True)


if __name__ == "__main__":
    main()
