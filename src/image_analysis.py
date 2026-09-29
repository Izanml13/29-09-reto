"""Parte 2 — Análisis de imágenes con Gemini.

Cuatro tareas por imagen, cada una con su prompt y esquema JSON:
OCR, descripción, clasificación de escena y detección de objetos.
Sigue el patrón de ``descripcion_imagen()``: se envía ``[prompt, PIL.Image]``.

Uso:
    python -m src.image_analysis                          # 5 imágenes x 4 tareas
    python -m src.image_analysis --tareas ocr escena
    python -m src.image_analysis --imagenes data/imagenes/paisaje.jpg https://.../foto.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils import (
    DATA_DIR, RESULTS_DIR, ErrorAPIKey, cargar_imagen, get_client, guardar_json, llamar_gemini_json,
)

# 5 imágenes de prueba + ejemplo.jpg heredada del repo 28-09-Reto (en la raíz)
IMAGENES_DEFECTO = [
    DATA_DIR / "imagenes" / n
    for n in ["doc_escaneado.png", "foto_urbana.jpg", "producto.jpg", "paisaje.jpg", "screenshot.png"]
] + [DATA_DIR.parent / "ejemplo.jpg"]

ESCENAS = ["documento", "calle urbana", "producto", "paisaje de montaña", "captura de pantalla",
           "ilustración", "interior", "playa", "otro"]

TAREAS = {
    "ocr": {
        "prompt": (
            "Transcribe EXACTAMENTE todo el texto visible en la imagen, respetando mayúsculas, "
            "tildes, signos y números. Usa un salto de línea (\\n) por cada línea de texto, en orden "
            "de lectura (arriba-abajo, izquierda-derecha). No corrijas ni resumas. Si no hay texto, "
            'devuelve cadena vacía. Devuelve SOLO JSON: {"texto": "..."}'
        ),
        "schema": {"type": "object", "properties": {"texto": {"type": "string"}}, "required": ["texto"]},
        "max_tokens": 4096,
    },
    "descripcion": {
        "prompt": (
            "Describe la imagen en español en 2-3 frases objetivas. Después enumera entre 3 y 8 "
            "elementos clave visibles como sustantivos cortos en español (p. ej. \"coches\", \"montañas\"). "
            'Devuelve SOLO JSON: {"descripcion": "...", "elementos_clave": ["..."]}'
        ),
        "schema": {
            "type": "object",
            "properties": {
                "descripcion": {"type": "string"},
                "elementos_clave": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["descripcion", "elementos_clave"],
        },
        "max_tokens": 1024,
    },
    "escena": {
        "prompt": (
            "Clasifica el tipo de escena de la imagen en UNA de estas categorías: "
            + ", ".join(ESCENAS)
            + '. Indica tu confianza entre 0.0 y 1.0. Devuelve SOLO JSON: {"escena": "...", "confianza": 0.0}'
        ),
        "schema": {
            "type": "object",
            "properties": {"escena": {"type": "string", "enum": ESCENAS}, "confianza": {"type": "number"}},
            "required": ["escena", "confianza"],
        },
        "max_tokens": 256,
    },
    "objetos": {
        "prompt": (
            "Detecta los objetos físicos o elementos de interfaz distinguibles en la imagen y cuenta "
            "cuántos hay de cada uno. Usa nombres en español, en singular y genéricos (\"coche\", "
            "\"persona\", \"árbol\", \"botón\"). No incluyas el fondo ni el cielo. "
            'Devuelve SOLO JSON: {"objetos": [{"nombre": "...", "cantidad": N}]}'
        ),
        "schema": {
            "type": "object",
            "properties": {
                "objetos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"nombre": {"type": "string"}, "cantidad": {"type": "integer"}},
                        "required": ["nombre", "cantidad"],
                    },
                }
            },
            "required": ["objetos"],
        },
        "max_tokens": 1024,
    },
}


def ejecutar_tarea(client, imagen, tarea: str) -> dict:
    """Ejecuta una tarea visual (temperature=0: todas son extracción/clasificación)."""
    cfg = TAREAS[tarea]
    return llamar_gemini_json(
        client, [cfg["prompt"], imagen], temperature=0.0,
        max_output_tokens=cfg["max_tokens"], schema=cfg["schema"],
    )


def analizar_imagen(client, origen: str, tareas: list[str]) -> dict:
    """Carga una imagen (ruta o URL) y ejecuta sobre ella las tareas indicadas."""
    img = cargar_imagen(origen)
    resultado = {"imagen": Path(origen).name, "origen": str(origen), "tamano": list(img.size)}
    for t in tareas:
        print(f"   · {t}...", end=" ", flush=True)
        try:
            resultado[t] = ejecutar_tarea(client, img, t)
            print("ok")
        except ErrorAPIKey:
            raise
        except Exception as e:
            resultado[t] = {"error": str(e)}
            print(f"error: {e}")
    return resultado


def main() -> int:
    """CLI del análisis de imágenes."""
    p = argparse.ArgumentParser(description="Análisis de imágenes con Gemini")
    p.add_argument("--imagenes", nargs="*", help="Rutas o URLs (por defecto las 5 de data/imagenes)")
    p.add_argument("--tareas", nargs="*", choices=list(TAREAS), default=list(TAREAS))
    p.add_argument("--salida", default=str(RESULTS_DIR / "resultados_imagenes.json"))
    args = p.parse_args()

    imagenes = args.imagenes or [str(r) for r in IMAGENES_DEFECTO]
    try:
        client = get_client()
        resultados = []
        for i, origen in enumerate(imagenes, 1):
            print(f"[{i}/{len(imagenes)}] {origen}")
            try:
                resultados.append(analizar_imagen(client, origen, args.tareas))
            except (FileNotFoundError, OSError) as e:
                print(f"   ✗ no se pudo cargar: {e}", file=sys.stderr)
    except ErrorAPIKey as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    guardar_json(resultados, Path(args.salida))
    print(f"✓ Guardado en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
