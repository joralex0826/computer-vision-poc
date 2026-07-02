"""Utilidades compartidas."""
import json
import re


def parse_output(text):
    """Saca el primer JSON valido de la respuesta del modelo.
    Retorna None si no encuentra nada parseable.
    """
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def norm(text):
    """Minusculas y espacios colapsados, para comparar keywords de forma robusta."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def clean_ocr(text):
    """Colapsa saltos de linea y espacios repetidos a un solo espacio.

    Conserva mayusculas y acentos porque son senal util para el LLM.
    El match de keywords usa norm() por separado.
    """
    return re.sub(r"\s+", " ", str(text)).strip()
