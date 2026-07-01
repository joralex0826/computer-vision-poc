# Moderación de imágenes — POC

Prueba de concepto de un sistema que detecta imágenes que infringen la política comercial del marketplace (promociones, promesas de entrega y sellos de campaña sobrepuestos), reemplazando el flujo legacy de OCR + Regex. El objetivo es subir la precisión sin sacrificar el recall, con inferencia local (sin APIs externas de OCR) e interpretable.

La función de entrada es:

```python
from moderation import ModerationPipeline
pipe = ModerationPipeline()
pipe.moderate(picture_url="https://http2.mlstatic.com/....webp")
# -> {"has_infraction": bool, "evidence": str, "reason": ..., "stage": ..., "latency_ms": ...}
```

## Arquitectura

Cascada de bajo costo. El texto enruta (barato); la imagen decide (VLM, solo el residuo).

```
imagen
  -> OCR local (RapidOCR)     ¿hay texto?            no -> LIMPIO
  -> router (LLM de texto)    ¿promo/delivery/badge? no -> LIMPIO
  -> VLM (Qwen2.5-VL)         decide overlay-vs-fisico + evidencia
     [-> verificador opcional, etapa 2 de precision]
```

Todo corre on-prem (MLX en Apple Silicon / GPU): no hay dependencias de terceros en el camino de inferencia.

## Estructura

```
poc_ml/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- anexo_1_dataset.csv          # dataset (Anexo 1)
|   `-- golden/                      # golden set, imagenes y predicciones
|-- src/moderation/                  # paquete reutilizable
|   |-- config.py                    # rutas y parametros (modelos, umbrales)
|   |-- prompts.py                   # prompts (VLM, verificador, router)
|   |-- utils.py                     # parseo JSON, normalizacion, limpieza OCR
|   |-- ocr.py / router.py / vlm.py  # etapas del pipeline
|   |-- pipeline.py                  # ModerationPipeline.moderate()
|   `-- evaluation.py                # metricas (P/R/F1, Wilson, reponderacion)
|-- scripts/                         # jobs ejecutables
|   |-- run_vlm_annotation.py        # pre-etiquetado del golden con el VLM
|   |-- eval_engine.py               # evalua el VLM solo vs golden
|   |-- verify_stage.py              # cascada de verificacion (etapa 2)
|   `-- eval_pipeline.py             # evalua el pipeline completo end-to-end
|-- notebooks/
|   |-- eda.ipynb                    # analisis exploratorio
|   |-- golden_set.ipynb             # construccion y revision del golden set
|   `-- pipeline.ipynb               # demo visual de moderate()
|-- tests/                           # tests de funciones puras
`-- docs/
    |-- reporte.md                   # DOCUMENTO DE ENTREGA (solucion completa)
    `-- annotation_guide.md          # anexo: guia de anotacion del golden
```

El entregable que pide el reto es un solo documento: **[docs/reporte.md](docs/reporte.md)**. La guia de anotacion es un anexo de apoyo.

## Setup

Entorno: conda `computer-vision-poc` (Python 3.11, Apple Silicon).

```bash
pip install -r requirements.txt
```

## Uso

Demo visual (recomendado): abrir `notebooks/pipeline.ipynb` con el kernel `computer-vision-poc`.

Desde Python:

```python
import sys; sys.path.insert(0, "src")
from moderation import ModerationPipeline
pipe = ModerationPipeline(vlm_model="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
print(pipe.moderate(picture_url="..."))
```

Evaluación contra el golden:

```bash
# pipeline completo (override del VLM por entorno; es reanudable)
VLM_MODEL=mlx-community/Qwen2.5-VL-7B-Instruct-4bit python scripts/eval_pipeline.py all
```

Tests:

```bash
pytest tests/
```

## Resultados (golden set, n=150, VLM 7B)

| Sistema | Precisión | Recall | F1 |
|---|---|---|---|
| Legacy (OCR + Regex) | 0.53 | 1.00 | 0.70 |
| Pipeline (este trabajo) | 0.64 | 0.98 | 0.78 |

El detalle, la metodología y las limitaciones están en [docs/reporte.md](docs/reporte.md).
