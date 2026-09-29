"""Parte 1 — Análisis de texto con Gemini.

Para cada texto de ``data/textos.jsonl`` (los 10 textos de TEXTOS_ES + TEXTOS_EN del
repo 28-09-Reto, ids 1-10) y de la ampliación (ids 11-20) obtiene idioma, entidades tipadas, sentimiento y resumen.

Se hacen dos llamadas por texto para respetar la reproducibilidad pedida:
- extracción (idioma, entidades, sentimiento) con temperature=0
- resumen con temperature=0.3

Uso:
    python -m src.text_analysis                 # todos los textos
    python -m src.text_analysis --ids 1 4 9     # solo algunos
    python -m src.text_analysis --tarea extraccion   # sin resumen
"""
from __future__ import annotations

import argparse
import sys

from src.utils import (
    DATA_DIR, RESULTS_DIR, ErrorAPIKey, get_client, guardar_json, leer_jsonl, llamar_gemini_json,
)

TIPOS_ENTIDAD = ["PERSONA", "ORGANIZACION", "UBICACION", "FECHA", "OTRO"]

# Evolución de PROMPT_SENTIMIENTO (main.py): añade idioma, entidades tipadas y reglas explícitas.
PROMPT_EXTRACCION = """Eres un sistema de análisis lingüístico. Analiza el TEXTO y devuelve SOLO un JSON con este esquema exacto:
{{
  "idioma": "código ISO 639-1 del idioma dominante (es, en, pt, ...)",
  "entidades": [{{"texto": "mención literal tal como aparece", "tipo": "PERSONA|ORGANIZACION|UBICACION|FECHA|OTRO"}}],
  "sentimiento": {{"polaridad": "positivo|negativo|neutro", "confianza": número entre 0.0 y 1.0}}
}}

Reglas:
- idioma: si el texto mezcla idiomas, devuelve el que aporta la estructura gramatical principal.
- entidades: copia la mención literal (sin traducir). PERSONA = personas (incluye tratamientos como "Sr."), ORGANIZACION = empresas, instituciones, departamentos, UBICACION = ciudades/países/lugares, FECHA = fechas y expresiones temporales ("ayer", "Monday"), OTRO = productos, marcas de producto, cantidades de dinero. Una fecha con su hora es UNA sola entidad ("Monday at 10am"). Las duraciones ("tres semanas", "5 horas", "dos años") NO son entidades. Los nombres comunes genéricos ("la compañía", "el restaurante") NO son entidades. No repitas entidades.
- sentimiento: evalúa la intención del autor. Detecta sarcasmo e ironía (elogios exagerados ante algo negativo => negativo). Las noticias y emails informativos sin carga emocional son "neutro". Si hay aspectos positivos y negativos equilibrados, usa "neutro".
- confianza: tu seguridad en la polaridad.

TEXTO:
\"\"\"{texto}\"\"\""""

PROMPT_RESUMEN = """Resume el TEXTO en UNA frase de máximo 25 palabras, en el mismo idioma del texto, conservando las entidades clave (personas, organizaciones, lugares, cifras).
Devuelve SOLO JSON: {{"resumen": "..."}}

TEXTO:
\"\"\"{texto}\"\"\""""

SCHEMA_EXTRACCION = {
    "type": "object",
    "properties": {
        "idioma": {"type": "string"},
        "entidades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "texto": {"type": "string"},
                    "tipo": {"type": "string", "enum": TIPOS_ENTIDAD},
                },
                "required": ["texto", "tipo"],
            },
        },
        "sentimiento": {
            "type": "object",
            "properties": {
                "polaridad": {"type": "string", "enum": ["positivo", "negativo", "neutro"]},
                "confianza": {"type": "number"},
            },
            "required": ["polaridad", "confianza"],
        },
    },
    "required": ["idioma", "entidades", "sentimiento"],
}


def extraer_info(client, texto: str) -> dict:
    """Detecta idioma, entidades y sentimiento (temperature=0, max 2048 tokens)."""
    return llamar_gemini_json(
        client, [PROMPT_EXTRACCION.format(texto=texto)],
        temperature=0.0, max_output_tokens=2048, schema=SCHEMA_EXTRACCION,
    )


def resumir(client, texto: str) -> str:
    """Genera un resumen de una frase (temperature=0.3)."""
    r = llamar_gemini_json(client, [PROMPT_RESUMEN.format(texto=texto)], temperature=0.3, max_output_tokens=2048)
    return r.get("resumen", "") if isinstance(r, dict) else str(r)


def analizar_texto(client, item: dict, con_resumen: bool = True) -> dict:
    """Analiza un registro de textos.jsonl y devuelve el JSON del esquema pedido."""
    info = extraer_info(client, item["texto"])
    sent = info.get("sentimiento", {}) or {}
    return {
        "id": item["id"],
        "tipo": item.get("tipo"),
        "subtipo": item.get("subtipo"),
        "idioma": str(info.get("idioma", "")).lower()[:2],
        "texto": item["texto"],
        "entidades": [e for e in info.get("entidades", []) if e.get("tipo") in TIPOS_ENTIDAD],
        "sentimiento": {
            "polaridad": str(sent.get("polaridad", "")).lower(),
            "confianza": float(sent.get("confianza", 0.0)),
        },
        "resumen": resumir(client, item["texto"]) if con_resumen else "",
    }


def main() -> int:
    """CLI del análisis de texto."""
    p = argparse.ArgumentParser(description="Análisis de texto con Gemini")
    p.add_argument("--entrada", default=str(DATA_DIR / "textos.jsonl"))
    p.add_argument("--salida", default=str(RESULTS_DIR / "resultados_texto.json"))
    p.add_argument("--ids", type=int, nargs="*", help="Procesar solo estos ids")
    p.add_argument("--tarea", choices=["todo", "extraccion"], default="todo")
    args = p.parse_args()

    items = leer_jsonl(args.entrada)
    if args.ids:
        items = [i for i in items if i["id"] in args.ids]

    try:
        client = get_client()
    except ErrorAPIKey as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    resultados = []
    for item in items:
        print(f"[{item['id']:>2}/{len(items)}] {item['tipo']}/{item['subtipo']}: {item['texto'][:50]}...")
        try:
            r = analizar_texto(client, item, con_resumen=args.tarea == "todo")
            print(f"     idioma={r['idioma']} sent={r['sentimiento']['polaridad']} entidades={len(r['entidades'])}")
        except ErrorAPIKey as e:
            print(f"❌ {e}", file=sys.stderr)
            return 1
        except Exception as e:  # un texto fallido no aborta el lote
            print(f"     ✗ error: {e}", file=sys.stderr)
            r = {"id": item["id"], "texto": item["texto"], "error": str(e)}
        resultados.append(r)

    from pathlib import Path
    guardar_json(resultados, Path(args.salida))
    print(f"✓ Guardado en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
