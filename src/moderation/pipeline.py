"""Pipeline completo de moderacion (POC end-to-end).

Funcion de entrada del reto: moderate(picture_url) -> {has_infraction, evidence}.
Cascada de bajo costo: OCR local -> router (texto) -> VLM (vision).
La interpretabilidad se cumple en cualquier etapa: la evidencia justifica la
decision con lo que esa etapa observo.
"""
import hashlib
import time
from pathlib import Path

from .config import CACHE_DIR, MODEL
from .ocr import OCRStage
from .router import RouterStage
from .vlm import Moderator


class ModerationPipeline:
    def __init__(self, vlm_model=MODEL):
        self.ocr = OCRStage()
        self.router = RouterStage()
        self.vlm = Moderator(model=vlm_model)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _resolve_image(self, picture_url, image_path):
        if image_path is not None:
            return Path(image_path)
        import requests
        dest = CACHE_DIR / (hashlib.md5(picture_url.encode()).hexdigest() + ".jpg")
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        tmp = dest.with_suffix(".tmp")
        try:
            r = requests.get(picture_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            tmp.write_bytes(r.content)
            if tmp.stat().st_size == 0:
                raise ValueError(f"Imagen vacia descargada desde {picture_url}")
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return dest

    def moderate(self, picture_url=None, image_path=None):
        t = {}
        s = time.perf_counter()
        path = self._resolve_image(picture_url, image_path)
        t["resolve_ms"] = (time.perf_counter() - s) * 1000

        s = time.perf_counter()
        text = self.ocr(path)
        t["ocr_ms"] = (time.perf_counter() - s) * 1000

        if not text.strip():
            return {"has_infraction": False,
                    "evidence": "El OCR local no detecto texto en la imagen, por lo que no hay overlay promocional, de entrega ni sello sobrepuesto.",
                    "reason": "none", "stage": "ocr", "ocr_text": "",
                    "latency_ms": round(sum(t.values()), 1), "stage_ms": t}

        s = time.perf_counter()
        suspicious = self.router.is_suspicious(text)
        t["router_ms"] = (time.perf_counter() - s) * 1000

        if not suspicious:
            return {"has_infraction": False,
                    "evidence": f'El texto detectado no contiene lenguaje de promocion, envio/entrega ni sellos de plataforma. Texto OCR: "{text[:160]}"',
                    "reason": "none", "stage": "router", "ocr_text": text,
                    "latency_ms": round(sum(t.values()), 1), "stage_ms": t}

        s = time.perf_counter()
        result = self.vlm.moderate(path)
        t["vlm_ms"] = (time.perf_counter() - s) * 1000
        result["stage"] = "vlm"
        result["ocr_text"] = text
        result["latency_ms"] = round(sum(t.values()), 1)
        result["stage_ms"] = t
        return result
