"""Sistema de moderacion de imagenes para deteccion de infracciones de politica.

API publica:
    from moderation import ModerationPipeline, Moderator
    pipe = ModerationPipeline()
    pipe.moderate(picture_url="https://...")  # -> {has_infraction, evidence, ...}
"""
from .config import MODEL, ROUTER_MODEL
from .evaluation import evaluate, precision_at_prevalence, wilson_ci
from .ocr import OCRStage
from .pipeline import ModerationPipeline
from .prompts import PROMPT, ROUTER_PROMPT, VERIFY_PROMPT
from .router import RouterStage
from .utils import parse_output
from .vlm import Moderator

__all__ = [
    "ModerationPipeline", "Moderator", "OCRStage", "RouterStage",
    "MODEL", "ROUTER_MODEL", "PROMPT", "VERIFY_PROMPT", "ROUTER_PROMPT",
    "parse_output", "evaluate", "wilson_ci", "precision_at_prevalence",
]
