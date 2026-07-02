"""Etapa de ruteo: decide si el texto OCR amerita revision visual (VLM).

El LLM cubre parafrasis/multilingue; los keywords cubren
terminos fijos que el modelo chico podria pasar por alto.
"""
from .config import ROUTER_MODEL
from .prompts import ROUTER_KEYWORDS, ROUTER_PROMPT
from .utils import clean_ocr, norm


class RouterStage:
    """Filtro de texto barato que decide si vale la pena llamar al VLM.

    Dos capas en paralelo: keywords exactos (rapido) + LLM 1.5B (parafrasis).
    """

    def __init__(self, model=ROUTER_MODEL):
        from mlx_lm import load
        self.model, self.tok = load(model)

    def is_suspicious(self, text):
        """Retorna True si el texto suena a promo, delivery o badge de plataforma.

        Primero revisa keywords si no hay match, consulta
        el LLM. Asi se evita la latencia del modelo en los casos mas obvios.
        """
        n = norm(text)
        if any(k in n for k in ROUTER_KEYWORDS):
            return True
        from mlx_lm import generate
        msgs = [{"role": "user", "content": ROUTER_PROMPT.format(t=clean_ocr(text)[:800])}]
        p = self.tok.apply_chat_template(msgs, add_generation_prompt=True)
        out = generate(self.model, self.tok, prompt=p, max_tokens=4, verbose=False)
        return "REVISAR" in out.upper()
