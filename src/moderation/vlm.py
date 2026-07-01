"""Etapa de vision: VLM local (Qwen2.5-VL via MLX) que decide y justifica."""
from .config import MODEL, VLM_MAX_SIDE, VLM_MAX_TOKENS
from .prompts import PROMPT
from .utils import parse_output


class Moderator:
    """Motor de moderacion basado en VLM local.

    moderate(image_path) -> {has_infraction: bool, evidence, reason, confidence}.
    El prompt es configurable para reutilizar el mismo motor como verificador.
    """

    def __init__(self, model=MODEL, max_tokens=VLM_MAX_TOKENS, prompt=None, max_side=VLM_MAX_SIDE):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model, self.processor = load(model)
        self.config = load_config(model)
        self.max_tokens = max_tokens
        self.prompt = prompt if prompt is not None else PROMPT
        self.max_side = max_side

    def _infer(self, image_path):
        import os
        import tempfile
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        path, tmp = str(image_path), None
        if max(img.size) > self.max_side:  # redimensiona para evitar timeouts de GPU
            img.thumbnail((self.max_side, self.max_side))
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            img.save(tmp, "JPEG", quality=90)
            path = tmp
        try:
            prompt = apply_chat_template(self.processor, self.config, self.prompt, num_images=1)
            out = generate(self.model, self.processor, prompt, image=[path],
                           max_tokens=self.max_tokens, verbose=False)
            text = out.text if hasattr(out, "text") else str(out)
            return parse_output(text) or {"has_infraction": False, "reason": "none",
                                          "evidence": "no_parse", "confidence": "low"}
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    def moderate(self, image_path):
        r = self._infer(image_path)
        return {"has_infraction": bool(r.get("has_infraction")),
                "evidence": str(r.get("evidence")),
                "reason": str(r.get("reason")),
                "confidence": str(r.get("confidence"))}
