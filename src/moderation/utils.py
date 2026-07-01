"""Utilidades compartidas."""
import json
import re


def parse_output(text):
    """Extrae el primer objeto JSON de la respuesta del modelo. None si falla."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def norm(text):
    """Normaliza espacios/saltos de linea a minusculas para el match de keywords."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def clean_ocr(text):
    """Limpia el texto del OCR antes de pasarlo al router/LLM.

    Colapsa saltos de linea, tabs y espacios repetidos a un solo espacio (evita
    ruido y tokens desperdiciados) y conserva mayusculas y acentos, que son senal
    util para el LLM. El match de keywords se hace aparte con norm().
    """
    return re.sub(r"\s+", " ", str(text)).strip()
