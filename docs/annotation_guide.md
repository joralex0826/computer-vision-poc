# Guia de anotacion: deteccion de infracciones en imagenes

Version 1.1. Criterio de etiquetado para construir el golden set del sistema de moderacion de imagenes.

## 1. Objetivo

Determinar, para cada imagen de producto, si **infringe la politica de contenido** del marketplace. La etiqueta resultante (`final_label`) es la verdad de referencia contra la que se mide el sistema (precision y recall >= 95%).

El anotador decide mirando la imagen, no la etiqueta del sistema legacy ni la salida del VLM (esas son solo apoyo).

## 2. Definicion de infraccion

Una imagen **infringe** si contiene **texto, sellos o graficos SOBREPUESTOS** (anadidos sobre la foto para llamar la atencion o simular funciones de la plataforma) que pertenezcan a alguna de estas tres categorias:

- **promo** — promociones o reclamos comerciales: descuentos, ofertas, precios, "gratis", "2x1", "liquidacion", cuotas o financiacion ("12 cuotas sin interes").
- **delivery** — promesas de envio o entrega: "envio gratis", "llega hoy", "llega manana", "entrega inmediata", "pronta entrega".
- **badge** — sellos que imitan etiquetas oficiales de MercadoLibre: "Mas vendido", "Recomendado", "Tienda Oficial", "Best Seller", "Hot Sale", "Mercado Lider", "Black Friday" y similares, dibujados sobre la imagen.

El elemento clave es que sea **sobrepuesto** (overlay): un grafico anadido en edicion, no parte fisica del producto.

## 3. Que NO es infraccion

- **Texto que es parte fisica del producto o su empaque.** Ejemplos: una caja de cereal que dice "FREE inside", una camiseta cuyo diseno estampado dice "Black Friday", un envase que trae impreso "2x1" de fabrica.
- **Overlays informativos no comerciales.** Ejemplos: guia de tallas, tabla de medidas, especificaciones tecnicas, instrucciones. No promocionan nada.
- **Marca o nombre de la tienda como marca de agua discreta**, sin reclamo comercial.
- **Fotos limpias de producto** sin texto anadido.

## 4. Regla de decision

```
0. La imagen NO muestra el producto (es un QR de pago, captura de pantalla, solo un logo de plataforma)?
   - Si -> INFRACCION (categoria: no-producto / redireccion fuera de plataforma)
1. Hay texto/grafico SOBREPUESTO digitalmente, es decir anadido en edicion,
   que NO sea parte fisica del producto NI de la escena real fotografiada?
   - No (es parte del producto, del empaque, o de la escena fisica real) -> NO infraccion
   - Si -> seguir
2. Ese overlay es de tipo promo, delivery o badge-de-MercadoLibre?
   - Si -> INFRACCION (asignar categoria; si aplica mas de una, marcar todas)
   - No (informativo: specs / guia de tallas; autenticidad o garantia;
     marca de agua de marca propia) -> NO infraccion
```

## 5. Casos borde (donde se gana o pierde el 95%)

| Caso | Decision | Nota |
|---|---|---|
| Camiseta estampada "Black Friday" como diseno del producto | NO infraccion | El texto es el producto, no un overlay |
| Banner "50% OFF" pegado en una esquina en edicion | Infraccion (promo) | Overlay comercial |
| Sello dibujado "MAS VENDIDO" estilo MercadoLibre | Infraccion (badge) | Imita etiqueta de plataforma |
| "ENVIO GRATIS" sobrepuesto en franja | Infraccion (delivery) | Promesa de entrega |
| Guia de tallas sobrepuesta | NO infraccion | Overlay informativo |
| Precio de fabrica impreso en el empaque | NO infraccion | Parte fisica |
| Sticker de precio promocional anadido sobre la foto | Infraccion (promo) | Overlay comercial |
| Texto promocional en otro idioma (ej. ingles, portugues) | Infraccion | El idioma no cambia la categoria |
| Texto diminuto o ilegible, sin overlay comercial claro | NO infraccion | Marcar confianza baja |
| Watermark con nombre de tienda + "ofertas" | Infraccion (promo) | Tiene reclamo comercial |
| Cartel/banner fisico en la escena real (concesionario, local) | NO infraccion | Texto fisico de la escena, no overlay digital |
| Sticker de precio fisico en parabrisas | NO infraccion | Parte fisica de la escena real |
| Overlay de specs tecnicas ("4 SFP", "POE IN 9-57V") | NO infraccion | Overlay informativo, no comercial |
| Reclamo de autenticidad/garantia ("100% ORIGINAL", "garantia total") | NO infraccion | No es promo/delivery/badge |
| Marca de agua de marca propia ("TARMOC Official Store") | NO infraccion | Marca propia, no imita sello de ML |
| Logo "Mercado Pago" impreso como sponsor en la prenda | NO infraccion | Parte fisica del producto |
| Sello "Mercado Livre Full / Pronta entrega" sobrepuesto | Infraccion (badge + delivery) | Sello de plataforma sobrepuesto |
| QR de pago / imagen que no muestra el producto | Infraccion (no-producto) | Redireccion fuera de plataforma |

## 5.1 Principios para zonas grises

Tres reglas que resuelven la mayoria de los casos confusos:

1. **Overlay digital vs escena fisica.** Solo cuenta como infraccion el grafico/sello/texto **anadido digitalmente** sobre la imagen. El texto que ya esta en la escena real fotografiada (carteles de un local, stickers de parabrisas, etiquetas de gondola) o impreso en el producto/empaque **no es overlay** y no es infraccion.
2. **Overlay no es lo mismo que overlay infractor.** Que haya un grafico anadido no basta. Solo es infraccion si es **promo, delivery o badge-de-MercadoLibre**. Los overlays informativos (specs tecnicas, guia de tallas), los reclamos de autenticidad/garantia y las marcas de agua de marca propia estan permitidos.
3. **Imagen no-producto.** Si la imagen no muestra el producto (QR de pago, captura de pantalla, solo el logo de una plataforma), es infraccion por una categoria aparte: no-producto / redireccion fuera de plataforma.

## 5.2 Idiomas

El marketplace opera en espanol y portugues. Los terminos de envio/promo cuentan en ambos idiomas. Ejemplos en portugues: "frete gratis" (envio gratis), "envio imediato", "envio rapido", "pronta entrega", "entrega imediata", "promocao", "desconto", "a vista", "parcelado". El idioma no cambia la categoria.

## 6. Multiples razones

Una imagen puede tener varias categorias a la vez (ej. "ENVIO GRATIS" + "50% OFF" = delivery + promo). Para el golden basta con `final_label = True`; en `human_reason` se pueden listar las categorias separadas por coma.

## 7. Proceso de etiquetado (dos roles)

- **Anotador 1 (VLM, Qwen2.5-VL-3B):** propone de forma automatica `vlm_has_infraction`, `vlm_reason`, `vlm_evidence`, `vlm_confidence` para las 700 imagenes.
- **Adjudicador (humano):** revisa con la herramienta del notebook (`golden_set.ipynb`, seccion 6) los casos que importan, bajo un esquema de agreement-based labeling:
  - conflictos donde el VLM discrepa de la etiqueta del sistema legacy,
  - el bucket donde VLM y legacy coinciden en INFRACCION (un spot-check mostro que ~37% de estos eran falsos positivos del sobre-marcado compartido, por lo que se revisa completo),
  - errores de parseo del VLM.

  El bucket donde VLM y legacy coinciden en "limpio" se toma como verdad (el spot-check mostro 0% de error). El humano fija `final_label`.

## 8. Confianza y dudas

- Si tras mirar la imagen no queda claro (texto ilegible, recorte ambiguo), preferir **NO infraccion** y registrar la duda en `notes`.
- Ante un caso nuevo no cubierto por esta guia, anotarlo en `notes` para refinar la guia en una proxima version.

## 9. Salida

Por cada imagen revisada:

- `final_label`: booleano. Verdad de referencia.
- `human_label`: booleano puesto por el adjudicador (solo en los casos revisados).
- `human_reason`: categorias aplicables (promo / delivery / badge), opcional.
- `notes`: observaciones de casos borde o dudas.
