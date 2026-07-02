from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
GOLDEN_DIR = DATA_DIR / "golden"
CACHE_DIR = DATA_DIR / "cache_images"

# Modelos locales. Los scripts permiten override por la variable de
# entorno VLM_MODEL (p. ej. el 7B para mejor calidad).
MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"        # VLM por defecto (POC)
ROUTER_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"  # LLM de texto del router

# Cap del lado mayor de la imagen antes del VLM evita timeouts de GPu
# por exceso de tokens visuales y acelera la inferencia.
VLM_MAX_SIDE = 1024
VLM_MAX_TOKENS = 200

# Prevalencia real de positivos tras deduplicar (para reponderar la precision).
REAL_PREVALENCE = 27256 / 489639
