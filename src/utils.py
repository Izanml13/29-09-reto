"""Utilidades compartidas: cliente Gemini, carga de imágenes, reintentos y validación JSON.

Extraído y ampliado a partir de ``main.py`` (repo 28-09-Reto):
- ``get_client()``      -> configuración del cliente con GEMINI_API_KEY
- ``cargar_imagen()``   -> imagen desde ruta local o URL
- manejo de errores 429 / API_KEY_INVALID -> ``llamar_gemini_json()`` con backoff exponencial
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "resultados"

load_dotenv(ROOT / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PAUSA_ENTRE_LLAMADAS = 1.0  # rate limiting: time.sleep(1) entre llamadas
MAX_REINTENTOS = 4


class ErrorAPIKey(RuntimeError):
    """La API key falta o es inválida: no tiene sentido reintentar."""


def get_client():
    """Crea el cliente de Gemini leyendo GEMINI_API_KEY del entorno / .env.

    Returns:
        google.genai.Client configurado.

    Raises:
        ErrorAPIKey: si la variable no está definida.
    """
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ErrorAPIKey(
            "Falta GEMINI_API_KEY. Copia .env.example a .env y añade tu clave "
            "(https://aistudio.google.com/apikey)."
        )
    return genai.Client(api_key=api_key)


def cargar_imagen(origen: str | Path) -> Image.Image:
    """Carga una imagen desde una ruta local o una URL http(s).

    Args:
        origen: ruta de fichero o URL.

    Returns:
        Imagen PIL en modo RGB.
    """
    origen = str(origen)
    if origen.startswith(("http://", "https://")):
        req = urllib.request.Request(origen, headers={"User-Agent": "reto-gemini/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            img = Image.open(io.BytesIO(resp.read()))
    else:
        ruta = Path(origen)
        if not ruta.is_absolute():
            ruta = ROOT / ruta
        if not ruta.exists():
            raise FileNotFoundError(f"No existe la imagen: {ruta}")
        img = Image.open(ruta)
    return img.convert("RGB")


def _es_error_clave(exc: Exception) -> bool:
    """Detecta errores de API key (no reintentables)."""
    msg = str(exc)
    return any(c in msg for c in ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "UNAUTHENTICATED", "401"))


def _es_error_reintentable(exc: Exception) -> bool:
    """Detecta errores transitorios: 429 (cuota), 500/503 (servidor), timeouts."""
    msg = str(exc)
    return any(c in msg for c in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "timed out", "DEADLINE"))


def extraer_json(texto: str) -> Any:
    """Parsea JSON del modelo tolerando vallas ```json ... ``` residuales.

    Raises:
        json.JSONDecodeError: si no es JSON válido.
    """
    limpio = texto.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", limpio, re.S)
    if m:
        limpio = m.group(1)
    return json.loads(limpio)


def llamar_gemini_json(
    client,
    contenidos: list,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    schema: dict | None = None,
) -> Any:
    """Llama a Gemini exigiendo JSON, con reintentos y corrección de JSON inválido.

    - Backoff exponencial (2, 4, 8, 16 s) ante 429/5xx.
    - Si la respuesta no es JSON válido, reenvía un prompt de corrección con la
      salida defectuosa.
    - Siempre espera ``PAUSA_ENTRE_LLAMADAS`` tras cada llamada (rate limiting).

    Args:
        client: cliente de ``get_client()``.
        contenidos: lista [prompt, (imagen PIL)...] como en ``descripcion_imagen()``.
        temperature: 0 para extracción/clasificación, 0.3 para resúmenes.
        max_output_tokens: límite de tokens de salida.
        schema: ``response_schema`` opcional (dict JSON Schema).

    Returns:
        Objeto Python resultante de ``json.loads``.
    """
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=schema,
    )
    actuales = list(contenidos)
    ultimo_error: Exception | None = None

    for intento in range(MAX_REINTENTOS):
        try:
            resp = client.models.generate_content(model=MODEL, contents=actuales, config=config)
            texto = resp.text or ""
            try:
                return extraer_json(texto)
            except json.JSONDecodeError as e:
                ultimo_error = e
                print(f"  ⚠ JSON inválido (intento {intento + 1}), pidiendo corrección...", file=sys.stderr)
                actuales = list(contenidos) + [
                    "Tu respuesta anterior no era JSON válido:\n"
                    f"{texto[:3000]}\n\n"
                    f"Error: {e}. Devuelve SOLO el JSON corregido, sin texto adicional ni ```."
                ]
        except Exception as e:  # errores de la API
            if _es_error_clave(e):
                raise ErrorAPIKey(
                    f"API key inválida o proyecto sin acceso ({MODEL}): {e}\n"
                    "Revisa la clave y el proyecto en https://aistudio.google.com/apikey, "
                    "o prueba otro modelo con GEMINI_MODEL en .env."
                ) from e
            if not _es_error_reintentable(e):
                raise
            ultimo_error = e
            espera = 2 ** (intento + 1)
            print(f"  ⚠ Error transitorio ({str(e)[:80]}...). Reintento en {espera}s", file=sys.stderr)
            time.sleep(espera)
        finally:
            time.sleep(PAUSA_ENTRE_LLAMADAS)

    raise RuntimeError(f"Fallo tras {MAX_REINTENTOS} intentos: {ultimo_error}")


def guardar_json(obj: Any, ruta: Path) -> None:
    """Escribe ``obj`` como JSON UTF-8 indentado, creando directorios."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def leer_jsonl(ruta: Path) -> list[dict]:
    """Lee un fichero JSON Lines."""
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
