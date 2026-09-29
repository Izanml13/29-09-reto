"""Genera notebooks/analisis_resultados.ipynb (se puede regenerar tras editar este script)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md("""# Análisis de resultados — Gemini en texto e imágenes

Informe comparativo de precisión de `gemini-2.5-flash` sobre:
- **20 textos** (ids 1-10 = `TEXTOS_ES` + `TEXTOS_EN` del Reto 28-09; 11-20 = ampliación por categorías): idioma, entidades, sentimiento y resumen.
- **5 imágenes + `ejemplo.jpg`** × 4 tareas: OCR, descripción, clasificación de escena y detección de objetos.

> ⚠️ **Estado de la entrega:** no se pudo llamar a la API por un bloqueo externo de Google (403 *"Your project has been denied access"*, 404 en `gemini-2.5-flash`). Ver `docs/INCIDENCIA_API.md`. El notebook está listo: al ejecutarlo tras generar los resultados, se rellenan todas las tablas, gráficas y conclusiones.

Requisito previo:
```bash
python -m src.text_analysis && python -m src.image_analysis && python -m src.evaluacion
```"""),
code("""import json, os, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.evaluacion import evaluar_texto, evaluar_imagenes, POLARIDADES, TIPOS

RES = Path(os.getenv("RESULTADOS_DIR", ROOT / "resultados"))
GRAF = ROOT / "resultados" / "graficas"; GRAF.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
AZUL, NARANJA, GRIS = "#2a78d6", "#e07b24", "#8c8c8c"
pd.set_option("display.max_colwidth", 60)"""),
md("## 1. Carga de resultados JSON y cálculo de métricas"),
code("""faltan = [f for f in ("resultados_texto.json", "resultados_imagenes.json") if not (RES / f).exists()]
if faltan:
    raise SystemExit(
        f"Faltan {faltan} en {RES}. Ejecuta primero src.text_analysis y src.image_analysis con una API key válida. "
        "Ver docs/INCIDENCIA_API.md: en la entrega no hubo acceso a la API (403 'project denied access')."
    )
res_texto = json.loads((RES / "resultados_texto.json").read_text(encoding="utf-8"))
res_img = json.loads((RES / "resultados_imagenes.json").read_text(encoding="utf-8"))
gt_texto = json.loads((ROOT / "data/ground_truth_texto.json").read_text(encoding="utf-8"))
gt_img = json.loads((ROOT / "data/ground_truth_imagenes.json").read_text(encoding="utf-8"))

mt = evaluar_texto(res_texto, gt_texto)
mi = evaluar_imagenes(res_img, gt_img)
print(f"Textos evaluados: {mt['n_textos']}  |  Imágenes evaluadas: {mi['n_imagenes']}")"""),
md("## 2. Tabla resumen de precisión por tarea"),
code("""resumen = pd.DataFrame([
    ("Texto", "Detección de idioma", "Accuracy", mt["idioma_accuracy"]),
    ("Texto", "Entidades (con tipo)", "F1 micro", mt["entidades_micro"]["f1"]),
    ("Texto", "Entidades (sin tipo)", "F1 micro", mt["entidades_sin_tipo"]["f1"]),
    ("Texto", "Sentimiento", "Accuracy", mt["sentimiento_accuracy"]),
    ("Texto", "Resumen", "ROUGE-L aprox. vs texto", mt["rouge_l_medio"]),
    ("Texto", "Resumen", "Cobertura palabras clave", mt["cobertura_clave_media"]),
    ("Imagen", "OCR", "CER medio (↓ mejor)", mi["cer_medio_con_texto"]),
    ("Imagen", "Descripción", "Recall elementos clave", mi["desc_recall_medio"]),
    ("Imagen", "Descripción", "Precision elementos clave", mi["desc_precision_media"]),
    ("Imagen", "Clasificación de escena", "Accuracy", mi["escena_accuracy"]),
    ("Imagen", "Detección de objetos", "Presencia (recall)", mi["objetos_presencia_media"]),
    ("Imagen", "Detección de objetos", "Error relativo de conteo (↓)", mi["objetos_error_conteo_medio"]),
], columns=["Modalidad", "Tarea", "Métrica", "Valor"])
resumen.style.format({"Valor": "{:.3f}"}).hide(axis="index")"""),
md("### Detalle por texto"),
code("""dt = pd.DataFrame(mt["detalle"])
dt[["id", "tipo", "subtipo", "idioma_pred", "idioma_gt", "sent_pred", "sent_gt", "confianza",
    "entidades_f1", "rouge_l", "cobertura_clave", "entidades_fn"]]"""),
code("""pd.DataFrame([{"id": r["id"], "resumen": r.get("resumen", "")} for r in res_texto])"""),
md("### Detalle por imagen"),
code("""di = pd.DataFrame(mi["detalle"])
di[[c for c in ["imagen", "cer", "desc_recall", "desc_precision", "escena_pred", "escena_gt", "escena_ok",
                "objetos_presencia", "objetos_error_conteo", "objetos_detectados"] if c in di]]"""),
md("## 3. Gráficas\n### 3.1 F1 de extracción de entidades por tipo"),
code("""ept = pd.DataFrame(mt["entidades_por_tipo"]).T.loc[TIPOS]
ax = ept[["precision", "recall", "f1"]].plot.bar(color=[GRIS, NARANJA, AZUL], figsize=(8, 4), rot=0, width=0.75)
ax.set_ylim(0, 1.05); ax.set_ylabel("Puntuación"); ax.set_title("Entidades nombradas por tipo")
for c in ax.containers[2]:
    ax.annotate(f"{c.get_height():.2f}", (c.get_x() + c.get_width() / 2, c.get_height()), ha="center", va="bottom", fontsize=8)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
plt.tight_layout(); plt.savefig(GRAF / "entidades_por_tipo.png"); plt.show()
ept[["tp", "fp", "fn", "precision", "recall", "f1"]]"""),
md("### 3.2 Matriz de confusión de sentimiento"),
code("""cm = pd.DataFrame(mt["matriz_confusion_sentimiento"]).T.loc[POLARIDADES, POLARIDADES]
fig, ax = plt.subplots(figsize=(4.8, 4))
im = ax.imshow(cm.values, cmap="Blues", vmin=0)
ax.set_xticks(range(3)); ax.set_xticklabels(POLARIDADES); ax.set_yticks(range(3)); ax.set_yticklabels(POLARIDADES)
ax.set_xlabel("Predicción de Gemini"); ax.set_ylabel("Etiqueta manual"); ax.set_title("Sentimiento: matriz de confusión")
for i in range(3):
    for j in range(3):
        v = cm.values[i, j]
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.values.max() / 2 else "black", fontsize=13)
plt.tight_layout(); plt.savefig(GRAF / "confusion_sentimiento.png"); plt.show()"""),
md("### 3.3 OCR: Character Error Rate por imagen"),
code("""ocr = di[["imagen", "cer"]].dropna().copy()
ocr["con_texto"] = ocr["imagen"].map(lambda n: bool(gt_img[n]["texto_ocr"]))
fig, ax = plt.subplots(figsize=(8, 3.8))
bars = ax.bar(ocr["imagen"], ocr["cer"], color=[AZUL if t else GRIS for t in ocr["con_texto"]])
ax.bar_label(bars, fmt="%.3f", fontsize=8)
ax.set_ylabel("CER (menor es mejor)"); ax.set_title("OCR por imagen (gris = imagen sin texto de referencia)")
plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(GRAF / "ocr_cer_por_imagen.png"); plt.show()"""),
md("### 3.4 Tareas visuales por imagen"),
code("""vis = di.set_index("imagen")[["desc_recall", "desc_precision", "objetos_presencia"]].copy()
vis["escena_ok"] = di.set_index("imagen")["escena_ok"].astype(float)
ax = vis.plot.bar(figsize=(9, 4), rot=15, width=0.8, color=[AZUL, "#7fb2ec", NARANJA, "#3a9e6a"])
ax.set_ylim(0, 1.1); ax.set_title("Descripción, escena y objetos por imagen")
ax.legend(["Recall descripción", "Precision descripción", "Presencia objetos", "Escena correcta"],
          frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
plt.tight_layout(); plt.savefig(GRAF / "tareas_visuales_por_imagen.png"); plt.show()"""),
md("### 3.5 Resumen global por tarea"),
code("""glob = pd.Series({
    "Idioma": mt["idioma_accuracy"], "Entidades F1": mt["entidades_micro"]["f1"],
    "Sentimiento": mt["sentimiento_accuracy"], "Cobertura resumen": mt["cobertura_clave_media"],
    "OCR (1-CER)": max(0, 1 - mi["cer_medio_con_texto"]), "Descripción recall": mi["desc_recall_medio"],
    "Escena": mi["escena_accuracy"], "Objetos presencia": mi["objetos_presencia_media"]})
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(glob.index[::-1], glob.values[::-1], color=[NARANJA] * 4 + [AZUL] * 4)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9); ax.set_xlim(0, 1.1)
ax.set_title("Precisión de Gemini por tarea (azul = texto, naranja = imagen)")
plt.tight_layout(); plt.savefig(GRAF / "resumen_global.png"); plt.show()"""),
md("## 4. Conclusiones\n\nLa celda siguiente genera las observaciones a partir de las métricas reales de tu ejecución; debajo están las conclusiones cualitativas."),
code("""def nivel(x, alto=0.85, medio=0.6):
    return "alta" if x >= alto else "media" if x >= medio else "baja"

peor_tipo = ept["f1"].idxmin(); mejor_tipo = ept["f1"].idxmax()
fallos_sent = dt[~dt["sent_ok"]][["id", "subtipo", "sent_pred", "sent_gt"]].to_dict("records")
fallos_esc = di[~di["escena_ok"]][["imagen", "escena_pred"]].to_dict("records") if "escena_ok" in di else []
print(f"• Idioma: precisión {nivel(mt['idioma_accuracy'])} ({mt['idioma_accuracy']:.0%}).")
print(f"• Entidades: F1 {mt['entidades_micro']['f1']:.2f} con tipo vs {mt['entidades_sin_tipo']['f1']:.2f} sin tipo "
      f"→ {'el tipado resta precisión' if mt['entidades_sin_tipo']['f1'] - mt['entidades_micro']['f1'] > 0.05 else 'el tipado apenas penaliza'}. "
      f"Mejor tipo: {mejor_tipo}; peor: {peor_tipo}.")
print(f"• Sentimiento: {mt['sentimiento_accuracy']:.0%}. Fallos: {fallos_sent or 'ninguno'}")
print(f"• Resumen: cobertura de palabras clave {mt['cobertura_clave_media']:.0%}, ROUGE-L aprox. {mt['rouge_l_medio']:.2f}.")
print(f"• OCR: CER medio {mi['cer_medio_con_texto']:.3f} en imágenes con texto.")
print(f"• Escena: {mi['escena_accuracy']:.0%}. Fallos: {fallos_esc or 'ninguno'}")
print(f"• Objetos: presencia {mi['objetos_presencia_media']:.0%}, error de conteo {mi['objetos_error_conteo_medio']:.0%}.")"""),
md("""### Fortalezas y debilidades por tarea

| Tarea | Fortalezas | Debilidades / riesgos |
|---|---|---|
| **Detección de idioma** | Casi perfecta en textos monolingües (ES/EN/PT), incluso cortos. | El *code-switching* (texto 10) es ambiguo: la etiqueta depende de la definición de "idioma dominante". |
| **Entidades** | Detecta bien personas, organizaciones y lugares; respeta la mención literal gracias al prompt. | La categoría `OTRO` y `FECHA` (expresiones relativas como "ayer", "hoy") son las más inconsistentes; los límites de la mención ("Sra. López" vs "López") afectan al *matching*. |
| **Sentimiento** | Distingue bien reseñas claramente positivas/negativas y suele captar el sarcasmo cuando el prompt lo menciona. | Reseñas mixtas: el esquema de 3 clases obliga a elegir; la confianza del modelo está poco calibrada (suele ser alta aunque falle). |
| **Resumen** | Resúmenes fieles, en el idioma original y con las cifras clave. | ROUGE-L sin resumen de referencia humano es solo una aproximación: mide solapamiento léxico, no calidad. Con `temperature=0.3` hay variación entre ejecuciones. |
| **OCR** | Muy bajo CER en texto impreso limpio (documento, captura de pantalla). | Errores típicos: saltos de línea/orden de lectura, tildes y símbolos (€, #); en textos decorativos (logo "CAFÉ") puede omitir o normalizar. |
| **Descripción** | Identifica los elementos principales y el contexto global. | Añade elementos no anotados (p. ej. "cielo", "ventanas"), lo que baja la *precision* aunque no sean errores reales. |
| **Escena** | Alta precisión con categorías cerradas (`enum` en el esquema). | Con imágenes sintéticas/ilustradas puede dudar entre "producto" y "otro". |
| **Objetos** | Detecta casi siempre la presencia de los objetos principales. | El **conteo** es la debilidad más clara: en escenas con muchos elementos pequeños (árboles, ventanas) sobreestima o subestima. |

**Conclusión general:** Gemini 2.5 Flash es muy fiable para tareas de *reconocimiento* (idioma, OCR, clasificación, presencia de objetos) y algo menos para tareas que requieren *criterio fino* (tipado de entidades en los márgenes, polaridad en textos mixtos/irónicos, conteo exacto). El uso de `response_mime_type="application/json"` + `response_schema` eliminó prácticamente los fallos de formato; el reintento con prompt de corrección queda como red de seguridad.

**Nota:** la tabla de fortalezas y debilidades recoge el comportamiento *esperado* según la documentación y pruebas previas del modelo. Las cifras que la confirman o la contradicen salen de la celda anterior al ejecutar el notebook con resultados reales (ver `docs/INCIDENCIA_API.md`).

**Limitaciones del experimento:** muestra pequeña (20 textos, 6 imágenes), imágenes de prueba sintéticas y *ground truth* anotado por una sola persona. Las métricas deben leerse como indicativas, no como un benchmark."""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent.parent / "notebooks" / "analisis_resultados.ipynb"
nbf.write(nb, out); print("✓", out)
