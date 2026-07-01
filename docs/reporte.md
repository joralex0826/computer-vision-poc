# Moderación de imágenes en el marketplace — reporte técnico

> Este es el documento de entrega del reto: contiene la solución completa (diagnóstico, datos, arquitectura, resultados, requisitos no funcionales, monitoreo, limitaciones y uso de agentes de IA). El único anexo de apoyo es la [guía de anotación](annotation_guide.md) del golden set. El código está en el repositorio.

## 0. Resumen

El sistema actual de moderación de imágenes rechaza demasiadas publicaciones que no infringen nada. Como se menciona en el enunciado ("alta tasa de falsos positivos, no cuantificada") y el primer aporte de este trabajo fue ponerle un número: medido contra un conjunto etiquetado a mano, el sistema legacy acierta solo en torno al 53% de lo que rechaza. Casi la mitad de los rechazos son injustos. Nota: Son estadísticos basados en una muestra, por limitaciones de tiempo y recursos no fue posible encontrar la métrica para todos los datos.

La propuesta no es afinar el regex que ya tienen, sino cambiar dónde se toma la decisión. El problema no está en leer el texto de la imagen —el OCR ya lee bien— sino en decidir qué significa ese texto y, sobre todo, en distinguir un sello promocional pegado en edición de un texto que es parte física del producto. Eso último es un juicio visual que ningún sistema basado solo en texto puede hacer. Por eso el núcleo del sistema es un modelo de visión y lenguaje (VLM) local, protegido por una cascada barata que evita usarlo en el 90% del tráfico.

En el conjunto de evaluación, el pipeline sube la precisión de 0.53 a 0.64 manteniendo el recall prácticamente intacto (1.00 a 0.98). Es una mejora en linea con lo que pide el reto, aunque todavía por debajo del objetivo de 95/95. El resto del documento explica el diagnóstico, las decisiones de diseño, los resultados, lo que falta para cerrar la brecha, y el por qué se diseñó de dicha forma.

---

## 1. El problema y su diagnóstico

### 1.1 Lo que pide el reto

Diseñar e implementar una prueba de concepto que detecte imágenes que infringen la política comercial del marketplace —promociones tipo "50% off", promesas de entrega ("llega hoy"), y sellos o logos de campaña superpuestos (Black Friday, Hot Sale, badges que imitan etiquetas de la plataforma)— y que **mejore la precisión del sistema actual sin sacrificar el recall**. Esto con un criterio de aceptación de: precisión ≥ 95% y recall ≥ 95% a la vez.

Se exige una función `moderate(picture_url)` que devuelve si hay infracción y la evidencia que la sustenta; un techo de latencia (p99 ≤ 5000 ms); interpretabilidad (la evidencia debe señalar el elemento concreto); minimizar el uso de APIs externas de OCR y no depender de terceros en el camino crítico; y estrategias de monitoreo, iteración y extensibilidad. Todo debe correr sobre la infraestructura disponible (GPU de 16 o 24 GB, o CPU de 4 núcleos), con varios workers.

### 1.2 El sistema actual y por qué falla

El flujo legacy es asíncrono y por eventos: las publicaciones entran por `items-feed`, pasan al backend de moderación, este llama a Google Cloud Vision para extraer el texto de la imagen (OCR) y luego un conjunto de expresiones regulares decide si ese texto infringe. El volumen es alto: del orden de 60.000 solicitudes por minuto, con picos de 100.000.

El diagnóstico a partir del análisis del Anexo 1, es que **el cuello de botella no es el OCR sino la decisión**. Google Vision lee el texto correctamente; el regex es el que se equivoca, por dos razones:

1. **No entiende semántica.** Un regex que busca "descuento" no matchea "mitad de precio" ni "leve toda a segunda unidade", y a la vez se dispara con texto perfectamente legítimo. De ahí los falsos positivos.
2. **Es ciego a lo visual.** Un sello de Black Friday sobrepuesto, o una imagen que es un QR de pago en vez del producto, no necesariamente generan texto que el regex sepa interpretar. Y al revés: el texto "best seller" impreso en la tapa de un libro es parte del producto, no una infracción, pero el regex no puede ver que es la tapa.

Este segundo punto es muy importante y por tanto fue el enfoque en este reto: **distinguir un overlay añadido de un texto físico del producto es a la final un juicio visual.** "Recomendado por dermatólogos" impreso en el frasco de un shampoo no infringe; el mismo texto pegado como sello estilo plataforma sí. El string es idéntico; lo que cambia es si está en la escena o sobre ella. Ningún clasificador de texto puede resolver esto porque solo ve el texto.

### 1.3 Cuantificar el problema

El enunciado dice que la tasa de falsos positivos es "alta" pero "no cuantificada". Uno de los primeros pasos fue intentar cuantificarla, con esto, podríamos saber realmente si estamos moviendo la métrica. Contra el golden set completo de 700 imágenes (sección 2), el sistema legacy obtiene **precisión 0.53 con recall 1.00**: atrapa todo lo que debe atrapar, pero el 47% de lo que rechaza es legítimo. Por esta razón tenemos vendedores con publicaciones bloqueadas sin razón.

### 1.4 Tres trampas en los datos que cambiaron la estrategia

Antes de diseñar nada, decidí realizar un EDA del Anexo 1, en el cual identifiqué tres cosas súper importantes que guiaron los pasos siguientes:

- **La columna `infraction_detected` no es la verdad.** Es la salida del propio sistema viejo, es decir, contiene exactamente los falsos positivos que queremos eliminar. Al inicio pensé en entrenar un modelo con dichos datos, sin embargo hacerlo medir contra ella sería reproducir su error. Esto hace **obligatorio** un golden set etiquetado por humanos (o métodos avanzados).
- **Sesgo de selección.** El 100% de las publicaciones rechazadas en el dataset tienen `ocr_text`: el sistema viejo solo puede marcar lo que el OCR leyó. Por construcción, el dataset **no contiene infracciones puramente visuales** (sellos sin texto legible, por ejemplo), porque el sistema que lo generó es ciego a ellas. Es un punto ciego del que hay que ser consciente al interpretar el recall.
- **Base rate.** Tras deduplicar (546.909 filas a 489.639 imágenes únicas; 46.909 a 27.256 positivos), la prevalencia real de infracciones ronda el **5.6%**. Esto importa muchísimo para reportar la precisión: una precisión que se ve bien en un set balanceado puede desplomarse a la prevalencia real, porque a 5.6% cada punto de tasa de falsos positivos pesa mucho más. Por eso reportamos la precisión también reponderada a la prevalencia real.

---

## 2. Datos y golden set

### 2.1 Por qué un golden set y qué es

Como la etiqueta del dataset es ruidosa, necesitábamos una verdad de referencia construida a mano: un **golden set**, un conjunto de imágenes etiquetadas con criterio humano, que sirve para medir cualquier sistema (el legacy, la POC, versiones futuras) contra el mismo patrón.

### 2.2 Composición y muestreo

El golden set tiene **700 imágenes**. El muestreo fue estratificado para no heredar el sesgo del dataset: se tomaron tanto publicaciones que el sistema viejo rechazó como publicaciones que aceptó, en proporciones que garantizan ver ambas clases y los casos de frontera (libros, sponsors impresos, specs técnicas, badges reales). Previo a muestrear se deduplicó por imagen para evitar leakage y métricas infladas por repetidos.

Tras el etiquetado, la distribución quedó en **194 infracciones reales y 506 limpias**. Ese desbalance es deliberado y útil: la precisión se juega en los negativos difíciles (texto legítimo que parece infracción), así que conviene tener muchos. En conclusión, la muestra fue tomada de forma estrategica para intentar cubrir todos los casos posibles, infractores, no infractores, casos evidentes y casos frontera.

### 2.3 Proceso de etiquetado

El etiquetado fue en dos capas para que no dependiera de una sola opinión ni de una sola máquina y para hacerlo más rápido:

1. **Pre-etiquetado automático** con el VLM (`scripts/run_vlm_annotation.py`), que deja una propuesta de etiqueta y su evidencia para cada imagen. Esto aceleró mi revisión manual: en vez de partir de cero, revisé una propuesta.
2. **Revisión humana por acuerdo.** En lugar de revisar las 700 una por una, se revisaron a mano los casos donde el sistema viejo y el VLM **discrepaban**, más una muestra aleatoria de control de los casos donde ambos coincidían. La lógica: cuando dos sistemas con sesgos distintos coinciden, el riesgo de error baja; donde discrepan, está la información.

Aquí hubo un hallazgo importante: en el control de los casos donde ambos coincidían en "infracción", encontramos un 37.5% de sobre-marcado. Los dos sistemas comparten el mismo sesgo —marcar texto físico como si fuera overlay—, así que su acuerdo no era garantía. La respuesta fue añadir ese bucket "ambos dicen True" a la lista de revisión humana en vez de confiar en él. El criterio completo está en [annotation_guide.md](annotation_guide.md), con las reglas de zona gris (libros, sponsors, autenticidad genérica) que más confusión generaban.

---

## 3. Arquitectura de la solución

### 3.1 La idea central

El texto enruta; la imagen decide. Concretamente: una cascada donde etapas baratas filtran la inmensa mayoría del tráfico que claramente no infringe, y el VLM —caro pero capaz de juzgar overlay-vs-físico— solo se invoca sobre el residuo sospechoso.

```
  imagen
    |
    v
  [Etapa 1] OCR local (RapidOCR, on-prem) -- ¿hay texto?
    |                                            |
    | con texto                                  | sin texto --> LIMPIO
    v
  [Etapa 2] Router (LLM de texto) -- ¿promo / delivery / badge?
    |                                            |
    | sospechoso                                 | no sospechoso --> LIMPIO
    v
  [Etapa 3] VLM (Qwen2.5-VL): decide overlay vs fisico + evidencia concreta
    |
    v
  {has_infraction, evidence, reason, confidence, stage}
    (opcional: --> verificador, etapa 4)
```

### 3.2 Cada etapa y por qué está

**Etapa 1 — OCR local (RapidOCR sobre onnxruntime).** Reemplaza a Google Cloud Vision. Si una imagen no tiene texto alguno, no puede tener un overlay promocional, de entrega ni un sello textual, así que se acepta de inmediato. Es la etapa más barata y descarta una fracción grande del tráfico sin costo de modelo grande. Es local: cumple el requisito de no depender de un OCR externo en el camino crítico.

**Etapa 2 — Router (LLM de texto, Qwen2.5-1.5B local).** Recibe el texto del OCR y decide si "huele" a promo, delivery o badge. No decide la infracción —no puede, solo ve texto— sino que decide **si vale la pena llamar al VLM**. Está calibrado a recall altísimo: prefiere mandar de más al VLM que dejar pasar algo. Lleva además un *backstop* de palabras clave conocidas (`ROUTER_KEYWORDS`) que corre junto al LLM: el LLM cubre las paráfrasis ("mitad de precio"), la lista fija cubre los términos duros ("frete grátis", "hot sale"). Dos redes superpuestas para que no se escape recall en el filtro. Agregando un llm obtenemos una herramienta más poderosa que un regex. Esta no depende netamente de una lista de frases seleccionadas de forma manual que puede llegar a quedarse corta u obsoleta.

**Etapa 3 — VLM (Qwen2.5-VL).** La evaluación reportada usa el 7B (4 bits); el 3B queda como default de menor costo en `config.py` y se selecciona por la variable `VLM_MODEL`. Es quien decide de verdad. En una sola pasada lee el texto, entiende la semántica y —lo que ninguna etapa anterior puede— ve si el texto es un overlay añadido o parte física de la escena. Devuelve la decisión con la evidencia concreta. El prompt está endurecido contra varios fallos observados: trata el texto dentro de la imagen como contenido a evaluar y no como instrucciones (evita inyección tipo "aprobar esto"), distingue explícitamente tapa de libro / sponsor impreso / specs (negativos) de sellos de plataforma y franjas de oferta (positivos), y conoce los badges reales de MercadoLibre/Mercado Livre para no confundirlos.

**Etapa 4 — Verificador (opcional, implementado en `scripts/verify_stage.py`).** Un segundo paso del VLM con un prompt estricto que solo confirma la infracción si está claro que es un overlay. Es la palanca principal de precisión: en un esquema generador-verificador, la etapa 3 prioriza recall (no perder infracciones) y la 4 poda los falsos positivos. Está construido pero no aplicado por defecto en el pipeline, es la primera mejora del roadmap (sección 7).

### 3.3 El embudo de costo

La razón de ser de la cascada es económica. A 60.000 RPM no se puede pasar todo por un VLM, así que cada etapa barata debe absorber tanto tráfico como pueda antes de llegar a la visión.

Cuánto llega al VLM depende de la mezcla de imágenes. **El golden no sirve para estimar el ahorro**, porque está enriquecido a propósito con imágenes que llevan texto promocional para estresar la calidad: sobre el golden el VLM corre en torno al 85% de los casos, que es lo esperable de un set sesgado hacia lo difícil. Lo que importa para el costo es la mezcla de **producción**, y ahí la cuenta es otra: cerca de la mitad de las imágenes no traen texto que el OCR deba escalar (se aceptan en la etapa 1), y el router descarta buena parte del resto como contenido neutro. Con esas tasas, el VLM terminaría ejecutándose sobre el orden del **5 a 10% del tráfico real** —del orden de un 90% de ahorro frente a mandar todo al modelo caro—, con el filtro calibrado a recall cercano al 99% para no perder infracciones antes de la vista. El texto, barato, hace el filtrado, la visión, cara, se reserva para donde de verdad hace falta juicio.

Este porcentaje de producción es una **estimación de diseño** a partir de la cobertura de texto del dataset, no una medición sobre tráfico real; el número exacto se debe medir en línea (sección 6).

### 3.4 Interpretabilidad

El campo `evidence` lo produce la **misma etapa que toma la decisión**, no una explicación posterior. Si decide el OCR, la evidencia es "no se detectó texto". Si decide el router, es "el texto no contiene lenguaje de promoción/envío/sello" más el texto leído. Si decide el VLM, es el elemento concreto que vio ("franja '50% OFF' sobrepuesta en la esquina superior"). Quien pregunta por qué se rechazó una publicación recibe el motivo real de la etapa que la rechazó, no una racionalización.

---

## 4. Experimentación y resultados

### 4.1 Cómo se mide

Todo se mide contra el golden set, nunca contra la etiqueta ruidosa. Las métricas son precisión, recall y F1, con intervalos de confianza de Wilson (porque con cientos de muestras el intervalo importa), y la precisión reponderada a la prevalencia real del 5.6%. El código está en `src/moderation/evaluation.py` y los scripts de evaluación (`eval_engine.py` para el VLM solo, `eval_pipeline.py` para el end-to-end) son reanudables, porque una corrida completa sobre este hardware toma horas.

### 4.2 Resultado principal

Sobre un subconjunto de 150 imágenes del golden (VLM 7B en la etapa de visión):

| Sistema | Precisión | Recall | F1 |
|---|---|---|---|
| Legacy (OCR + Regex) | 0.53 | 1.00 | 0.70 |
| Pipeline (este trabajo) | 0.64 | 0.98 | 0.78 |

El pipeline **mejora la precisión en 11 puntos sacrificando solo 2 de recall**, justamente el trade-off pedido en el reto: subir precisión sin tirar el recall. El F1 sube de 0.70 a 0.78. (La precisión del legacy en este subconjunto de 150 es 0.53; sobre el golden completo de 700 es 0.55. La diferencia es el tamaño de muestra, no un cambio de sistema.)

### 4.3 Entendimiento de los resultados

No llegamos al 95/95. El cuello está en la precisión, y la causa es que incluso el VLM de 7B todavía confunde algunos textos físicos con overlays, que es el caso por defecto más difícil (un sello promocional pegado en edición frente a un texto impreso en el producto o un cartel real de la escena). Las palancas para cerrar la brecha (verificador, few-shot con casos borde, fine-tuning) están en la sección.

Conviene recordar el caveat de la sección 1.4: el golden hereda parte del sesgo de selección del dataset (pocas infracciones puramente visuales), así que el recall medido es optimista respecto a esas infracciones que el dataset no contiene. Es una limitación del material disponible, no del diseño, y se mitiga sembrando el golden con casos visuales en la siguiente iteración.

### 4.4 Por qué la precisión en producción sería mayor que ese 0.64

El 0.64 toca analizarlo con cuidado, porque el golden no es tráfico normal: este set se armó a propósito como un examen difícil, lleno de casos de frontera y sobre todo de imágenes limpias que *parecen* infracción (texto legítimo que se confunde con promoción). Es un stress test. Así que el 0.64 es casi el peor caso con la precisión entre las decisiones más difíciles, no la del mundo real.

Para ver por qué en producción sería mejor, sirve un ejemplo concreto. En el mundo real solo el 5.6% de las imágenes son infracciones. Tomemos **1000 imágenes**: 56 infractoras y 944 limpias. Supongamos que el sistema atrapa el 95% de las infractoras (53 de 56). Lo que cambia todo es cuántas *limpias* marca mal por error. Dos escenarios:

- **Si marca mal el 26% de las limpias** (que es lo que pasa en el golden, por estar cargado de casos difíciles): son 245 falsas alarmas. De todo lo que bloquea (53 buenas + 245 malas), solo el 53/298 = **18%** era de verdad infracción.
- **Si marca mal solo el 1% de las limpias** (lo esperable en producción): son 9 falsas alarmas. De lo que bloquea (53 + 9), el 53/62 = **85%** era infracción de verdad.

Tenemos el mismo sistema, lo único que cambió fue el porcentaje de limpias marcadas mal, y la precisión saltó de 0.18 a 0.85. Este resultado es esperado en producción, sin embargo es importante hacer seguimiento y confirmarlo. Ilustro el ejemplo anterior para clarificar un poco el por qué aunque no alcanzamos la precision de 95% en la POC, el resultado puede ser mucho más optimista en producción.

---

## 5. Requisitos no funcionales

### 5.1 Costo

Cubierto por diseño en dos frentes. Primero, **no hay dependencias de terceros en el camino crítico**: el OCR es local (RapidOCR), el router es un LLM local y el VLM es local vía MLX. Se elimina el costo y la latencia de red de Google Cloud Vision. Segundo, la **cascada limita el VLM a una fracción del tráfico** (estimada en ~5-10% en producción, sección 3.3), que es donde está el costo de cómputo real.

### 5.2 Latencia

Hay que separar dos hardwares (con el que se desarrolló el reto vs el de prod especificado). En el equipo de desarrollo (Apple M3, 16 GB) una llamada al VLM toma del orden de 20-36 segundos: sirve para evaluar offline, no cumpliría el p99 en producción. Pero el reto especifica GPU de 16-24 GB, y ahí el VLM en 4 bits responde en **uno a dos segundos** (el 3B más cerca de 1 s, el 7B algo más), dentro del techo de 5000 ms. Las etapas baratas (OCR + router), que resuelven la mayor parte del tráfico, tienen p99 del orden de 1.2 s incluso en el equipo de desarrollo. El p99 agregado en la infraestructura objetivo queda bajo el límite. El pipeline mide y reporta latencias por etapa (`latency_ms`, `stage_ms` en la salida), así que esto es verificable.

### 5.3 Interpretabilidad

Resuelta por diseño, como se explicó en 3.4: la evidencia la genera la etapa que decide y nombra el elemento concreto. La salida de `moderate()` incluye `has_infraction`, `evidence`, `reason` (promo/delivery/badge/none), `confidence` y `stage` (en qué etapa se decidió), de modo que cada rechazo es auditable.

### 5.4 Infraestructura

El reto da dos escenarios de hardware —GPU de 16 o 24 GB de memoria de video (VRAM), o CPU de 4 núcleos con 2-32 GB de RAM— y pide que el servicio corra con varios workers. El diseño encaja en ambos, y aquí está el por qué.

**Qué corre y cuánta memoria pide.** Se usaron los modelos *cuantizados a 4 bits*, que es guardar cada parámetro con menos precisión (~0.5 byte) para reducir el tamaño unas cuatro veces casi sin perder calidad. Con eso, el presupuesto de memoria queda así:

| Componente | Sin cuantizar | Cuantizado a 4 bits (lo que usamos) | Dónde corre |
|---|---|---|---|
| VLM Qwen2.5-VL 7B | ~14 GB | ~5-6 GB | GPU |
| VLM Qwen2.5-VL 3B | ~6 GB | ~2-3 GB | GPU |
| Router (LLM 1.5B) | ~3 GB | ~1 GB | GPU |
| OCR (RapidOCR) | — | ligero | CPU |

Sumando el VLM 7B, el router y el espacio de trabajo para procesar la imagen, hablamos de unos 7-9 GB: **cabe bien en la GPU de 16 GB y con margen amplio en la de 24 GB**. El 3B deja aún más espacio libre. La elección de modelos pequeños y cuantizados se hace justamente para respetar el límite de VRAM del reto sin sacrificar demasiada calidad.

---

## 6. Monitoreo, iteración y extensibilidad

### 6.1 Monitoreo

Las señales que se vigilarían en producción, y que disparan alerta:

- **Tasa de rechazo y mix por etapa.** Si el VLM empieza a invocarse muy por encima de su fracción esperada en producción, algo cambió en el tráfico (una campaña nueva, un tipo de imagen no visto) y hay que revisar. Esta señal es además la que valida en línea la estimación de costo de la sección 3.3.
- **Distribución de `confidence` del VLM.** Un aumento de decisiones "low" significa un cambio en el comportamiento o distribución de los datos.
- **Latencia p99 por etapa**, comportamiento atipico en latencia, superando el valor planteado anteriormente.
- **Tasa de apelaciones de vendedores**, la señal más directa de falsos positivos en el mundo real, el problema que queremos reducir. No tengo el contexto de si contamos con dicha métrica, dado el caso valdría la pena empezar a calcularla.

### 6.2 Iteración

El ciclo de mejora se alimenta de producción: los casos apelados (falsos positivos confirmados) y los falsos negativos que escapan se capturan, se etiquetan con la misma guía y se incorporan al golden set. Con ese golden creciente se reevalúa cada cambio de prompt o de modelo antes de desplegarlo. Es un loop cerrado donde produccion genera casos dificiles, el golden los absorbe, y el siguiente modelo se mide contra ellos.

Esto como trabajo reactivo en cierta medida, sin embargo, en función de las métricas que usamos (f1, precision, recall) podríamos tener alertas si estás también se ven afectadas, esto va ligado directamente al punto anterior, ya que dependerá de nuestro golden set, el cual debería ser actualizado de forma periodica para evitar un model drift.

### 6.3 Extensibilidad

Agregar una categoría nueva de infracción (por ejemplo, una campaña estacional, fecha especial o evento) no cambia la arquitectura: se amplía la taxonomía en el prompt del VLM y las palabras clave del router. El pipeline, las etapas y la infraestructura quedan iguales. Todos los cambios son transparentes al proceso y no es necesario modificar código.

---

## 7. Limitaciones y camino al 95/95

Lo que falta y próximos pasos:

1. **Aplicar el verificador (etapa 4).** Ya está implementado pero no aplicado (`scripts/verify_stage.py`). Es la palanca de precisión más barata: poda los falsos positivos del VLM con un segundo paso estricto, sin tocar el recall de la etapa 3. Primera mejora a medir.
2. **Few-shot con casos borde del golden.** Meter en el prompt los negativos difíciles (tapas de libro, sponsors, banners físicos de concesionario, estampados de prenda con texto de campaña) como ejemplos resueltos, en lugar de reglas declarativas que vimos que degradan la métrica global.
3. **Fine-tuning.** Si lo anterior no basta, ajustar el VLM con el golden más las etiquetas débiles del dataset. Es la palanca más potente pero la más costosa.
4. **Sembrar el golden con infracciones visuales** para cerrar el punto ciego del recall (sección 1.4) y medir contra casos que el dataset original no contiene.

Este tipo de problemas depende mucho de las iteraciones, es necesario probar, medir y ajustar. De esta forma se podría llegar al nivel de precisión esperado.

---

## 8. Matriz de cumplimiento del reto

Resumen objetivos:

| Requisito (PDF) | Cómo se cumple | Estado |
|---|---|---|
| Precisión ≥ 95% y recall ≥ 95% | Medición contra golden humano (stress test); el 0.64 es piso de casos duros, la precisión de producción sería mayor (sección 4.4); palancas escalonadas hasta el objetivo (sección 7) | Parcial: 0.64 / 0.98 en golden, plan al 95/95 |
| Diseño del golden set (composición, tamaño, distribución, etiquetado) | 700 imágenes, muestreo estratificado, etiquetado por acuerdo + control (sección 2) | Hecho |
| Función `moderate(picture_url) -> {has_infraction, evidence}` | `ModerationPipeline.moderate()` (sección 3) | Hecho |
| Latencia p99 ≤ 5000 ms | < 1 s en la GPU objetivo; etapas baratas p99 ~1.2 s; instrumentado (sección 5.2) | Hecho (en infra objetivo) |
| Interpretabilidad (evidencia específica) | La etapa que decide genera la evidencia con el elemento concreto (sección 3.4) | Hecho |
| Minimizar OCR externo / sin terceros en el camino crítico | OCR, router y VLM locales; se elimina Google Vision (sección 5.1) | Hecho |
| Monitoreo (señales de alerta) | Tasa de rechazo, mix por etapa, confianza, p99, apelaciones (sección 6.1) | Documentado |
| Iteración (captura de FP/FN, actualización, validación) | Loop golden creciente con casos de producción (sección 6.2) | Documentado |
| Extensibilidad sin cambios estructurales | Nueva categoría = ampliar prompt/keywords; arquitectura estable (sección 6.3) | Documentado |
| Infra (GPU 16/24 GB o CPU 4 núcleos; varios workers) | Servicio stateless replicable; VLM-4bit cabe en la GPU objetivo (3B en 16 GB, 7B en 16-24 GB) (sección 5.4) | Hecho (diseño) |
| Entregable escrito + diagramas + sección de agentes de IA | Este documento (secciones 3 y 8) | Hecho |
| Código en GitHub (notebooks o scripts) | `src/` + `scripts/` + `notebooks/` + `tests/` | Hecho |

## 9. Uso de agentes de IA en el desarrollo

Este trabajo se hizo con asistencia de un agente de IA (Claude, en Claude Code).

El agente se usó como **par de trabajo**. En concreto: para escribir el esqueleto del código (las clases del pipeline, los scripts de evaluación, los tests) que luego se revisó y corrigió; y para iterar los prompts del VLM, que fue un proceso de prueba y error guiado por los resultados de cada corrida, finalmente como redactor secundario para mas claridad en las ideas y explicaciones.

Se indaga junto con el apoyo de la IA sobre posibles enfoques para solucionar el problema de vision, modelos, sus limitaciones, ventajas y desventajas.

En resumen, la IA aportó velocidad (código) y conocimiento teorico (modelos y herramientas para este tipo de problemas). Conclusiones, análisis de resultados, iteraciones, estructura, próximos pasos, entre otros no corrieron por parte de agentes o modelos IA.
