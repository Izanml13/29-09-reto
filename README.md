# Reto Gemini — Análisis avanzado de texto e imágenes

Evolución del Reto 28-09: se reutilizan `get_client()`, `cargar_imagen()`, el manejo de errores 429/`API_KEY_INVALID`, los 10 textos (`TEXTOS_ES` + `TEXTOS_EN`) y `ejemplo.jpg`. Ahora hay análisis de texto (idioma, entidades, sentimiento, resumen), análisis de imagen (OCR, descripción, escena, objetos) y una evaluación cuantitativa frente a un *ground truth* manual, con un informe en Jupyter.

> [!WARNING]
> **Estado: código completo y verificado, sin resultados reales todavía.**
> No se pudieron hacer las llamadas reales por un bloqueo externo de Google que no depende del proyecto:
> - `gemini-2.5-flash` → **404** *"no longer available to new users"*
> - resto de modelos → **403** *"Your project has been denied access. Please contact support."*
> - la única clave que llegó a responder fue invalidada en minutos (**401 UNAUTHENTICATED**)
>
> Detalle, diagnóstico y pasos para reanudar: **[docs/INCIDENCIA_API.md](docs/INCIDENCIA_API.md)**.
> Con una clave válida, basta con ejecutar los 4 comandos de la sección *Uso*. `resultados/` y las gráficas se generan solos.

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
cp .env.example .env    # y pon tu GEMINI_API_KEY
```

## Uso

```bash
python main.py --todo                        # demo rápida heredada (Reto 28-09)
python -m src.text_analysis                  # Parte 1 → resultados/resultados_texto.json
python -m src.image_analysis                 # Parte 2 → resultados/resultados_imagenes.json
python -m src.evaluacion                     # métricas → resultados/metricas.json
jupyter notebook notebooks/analisis_resultados.ipynb   # informe + gráficas en resultados/graficas/
```

Ejecución modular (argparse):

```bash
python -m src.text_analysis --ids 9 10 --tarea extraccion
python -m src.image_analysis --tareas ocr escena --imagenes data/imagenes/screenshot.png https://ejemplo.com/foto.jpg
python -m src.evaluacion --solo imagenes
```

Coste aproximado de una ejecución completa: 40 llamadas de texto + 24 de imagen (~3-4 min con la pausa de 1 s).

## Estructura

| Ruta | Contenido |
|---|---|
| `src/utils.py` | `get_client()`, `cargar_imagen()`, `llamar_gemini_json()` (backoff exponencial 2-4-8-16 s ante 429/5xx, reintento con prompt de corrección si el JSON es inválido, `time.sleep(1)` entre llamadas) |
| `src/text_analysis.py` | Prompt de extracción (idioma + entidades tipadas + sentimiento, `temperature=0`) y prompt de resumen (`temperature=0.3`); `response_schema` para forzar el JSON |
| `src/image_analysis.py` | Un prompt + esquema por tarea visual (`temperature=0`) |
| `src/evaluacion.py` | Accuracy, P/R/F1 por tipo, matriz de confusión, ROUGE-L aproximado, CER (Levenshtein), cobertura de elementos |
| `data/textos.jsonl` | ids 1-10: los textos originales de `main.py`; ids 11-20: ampliación con noticias ES/EN/PT, emails formales, reseña mixta, sarcasmo e idioma mixto (`scripts/generar_datos_texto.py`) |
| `data/ground_truth_*.json` | Etiquetas manuales (con notas en los casos dudosos y sinónimos aceptados) |
| `data/imagenes/` + `ejemplo.jpg` | 5 imágenes de prueba (`scripts/generar_imagenes.py`) + la imagen heredada |
| `notebooks/analisis_resultados.ipynb` | Tablas, 5 gráficas y conclusiones (regenerable con `scripts/crear_notebook.py`) |

## Esquema de salida (texto)

```json
{"id": 1, "idioma": "es", "texto": "...",
 "entidades": [{"texto": "Madrid", "tipo": "UBICACION"}],
 "sentimiento": {"polaridad": "neutro", "confianza": 0.9},
 "resumen": "..."}
```

## Decisiones de diseño y métricas

- **Dos llamadas por texto**: la extracción va a `temperature=0` y el resumen a `0.3`, como pide el requisito de reproducibilidad.
- **Matching de entidades**: difuso (normalizado, sin tildes, inclusión de subcadena), así "Sra. López" empareja con "López". Se da F1 con tipo y sin tipo para separar los errores de detección de los de clasificación.
- **Reseña mixta** → `neutro` (el esquema no tiene clase "mixto"). **Sarcasmo** → `negativo`.
- **ROUGE-L aproximado**: sin resúmenes de referencia humanos, se calcula el LCS entre el resumen y el texto original, más la **cobertura de palabras clave** anotadas.
- **CER** = distancia de Levenshtein / longitud de la referencia, con espacios colapsados. Puede superar 1 si el modelo inventa texto.
- **Imágenes sintéticas**: se generan con Pillow para que el *ground truth* sea exacto (texto OCR conocido, número de coches y personas, etc.). Para evaluar con fotos reales, sustitúyelas manteniendo el nombre y actualiza `ground_truth_imagenes.json`.

- **Modelo:** si `gemini-2.5-flash` devuelve 404 (retirado para cuentas nuevas), pon `GEMINI_MODEL=gemini-flash-latest` en `.env`.

---

# Reto 28-09 (original) — Gemini API con Python


## 1. Paso a paso realizado

### Paso 1 — Acceder a Google AI Studio
1. Ve a **https://aistudio.google.com** e inicia sesión con tu cuenta de Google.
2. En el panel izquierdo pulsa **Create prompt / Chat**.
3. Prueba un prompt, p. ej.: `Explícame en 3 frases qué es la API de Gemini`.
4. En la barra lateral derecha ajusta parámetros y observa el cambio:
   - **Temperature** (0.0–2.0): 0.2 = determinista (clasificar), 0.7–1.0 = creativo.
     Este reto usa `0.7` para texto libre, `0.2` para sentimiento, `0.6` para imagen.
   - **Top-P** (0–1): `0.9` en el script.
   - **Max output tokens**: `300` (texto), `2048` (sentimiento JSON), `800` (imagen).
   - Botón **Get code** te muestra el equivalente en Python (lo que replica `main.py`).

### Paso 2 — Generar API Key gratuita (sin tarjeta)
1. En AI Studio pulsa **Get API key** (arriba a la derecha).
2. **Create API key** > elige un proyecto de Google Cloud (crea uno si no tienes).
3. Copia la clave `AIza...`. Es gratis, no pide tarjeta.
4. Si se filtra: vuelve a **Get API key**, bórrala y crea otra.
5. Nunca la subas a git (este repo la ignora vía `.gitignore` + `.env`).

### Paso 3 — Documentación de inicio rápido
- Guía oficial: **https://ai.google.dev/gemini-api/docs/quickstart?lang=python**
- Patrón actual (`google-genai >= 2.0`):
  ```python
  from google import genai
  client = genai.Client()  # lee GEMINI_API_KEY del entorno
  resp = client.models.generate_content(
      model="gemini-2.5-flash",
      contents="Explain how AI works in a few words",
  )
  print(resp.text)
  ```

### Paso 4 — Autenticación y límites gratuitos
- **Auth:** la key viaja como header `x-goog-api-key`. El SDK la toma de
  `GEMINI_API_KEY` (o se pasa con `genai.Client(api_key="...")`).
- **Plan gratuito (Free tier, puede variar por región/modelo):**
  - `gemini-2.5-flash`: ~**15 RPM** (peticiones/min), ~**1 M tokens/día**.
  - Si ves `429 RESOURCE_EXHAUSTED`: espera 60 s (el script lo detecta y avisa).
  - Cuota y uso: AI Studio > **Usage / Get API key**.
  - Límites exactos: https://ai.google.dev/pricing

### Paso 5 — Instalar librería
```powershell
pip install -r requirements.txt
# equivale a: pip install google-genai Pillow python-dotenv
```

### Paso 6 — Configurar y ejecutar
```powershell
copy .env.example .env
notepad .env   # pega tu GEMINI_API_KEY

python main.py --todo              # todo: texto + sentimiento + imagen
python main.py --texto             # solo prueba de texto
python main.py --sentimiento       # solo sentimiento ES+EN
python main.py --imagen            # usa ejemplo.jpg (incluida)
python main.py --imagen ./mi.jpg   # tu propia foto
python main.py --imagen-url https://ejemplo.com/foto.jpg
```

## 2. Qué hace el script (`main.py`)

| Parte | Detalle |
|---|---|
| `prueba_texto()` | 1 prompt simple con `temperature=0.7, top_p=0.9, max_output_tokens=300` |
| `analisis_sentimiento()` | 5 ES + 5 EN, pide JSON con `sentimiento, confianza, entidades, justificacion`. Guarda `resultado_sentimiento.json`. Usa `temperature=0.2` + `response_mime_type="application/json"` |
| `descripcion_imagen()` | Envía `[prompt, PIL.Image]` al modelo. Guarda `resultado_imagen.txt`. Si no hay imagen, genera `ejemplo.jpg` con Pillow |

Textos analizados: ver `TEXTOS_ES` / `TEXTOS_EN` en `main.py`.

## 3. Resultados obtenidos (ejemplo)

> Para reproducir: `python main.py --todo` con tu key.

**Texto:** Gemini responde en 3 frases qué es la API (generación, multimodalidad, free tier).

**Sentimiento (resumen esperado):**
- ES1 Rosalía → positivo (entidades: Rosalía)
- ES2 compañía/tardaron → negativo
- ES3 calle Mayor 12 → neutro (entidades: calle Mayor 12)
- ES4 familia Madrid → positivo
- ES5 Iberia/5 horas → negativo
- EN1–EN5: positivo / negativo / neutro / positivo / negativo
- Fichero: `resultado_sentimiento.json` (se crea al ejecutar)

**Imagen:** con `ejemplo.jpg` describe sol, marco, texto visible y propone
usos (marketing, accesibilidad/alt-text, testing). Fichero: `resultado_imagen.txt`.

Sin key verás el mensaje de ayuda:
`ERROR: no hay GEMINI_API_KEY... Ve a https://aistudio.google.com...`

## 4. Estructura del entregable

```
main.py                    # script principal (texto + sentimiento + imagen)
requirements.txt           # google-genai, Pillow, python-dotenv
.env.example               # plantilla (copiar a .env)
ejemplo.jpg                # imagen de prueba
resultado_sentimiento.json # se genera al ejecutar
resultado_imagen.txt       # se genera al ejecutar
README.md                  # este fichero
```

## 5. Solución de problemas

| Error | Causa / fix |
|---|---|
| `no hay GEMINI_API_KEY` | No hay `.env`. Cópialo y pega la key |
| `API_KEY_INVALID` | Key mal copiada o borrada. Crea otra en AI Studio |
| `429 / RESOURCE_EXHAUSTED` | Límite 15 RPM. Espera 1 min, baja `max_output_tokens` |
| `404 model not found` | Cambia `GEMINI_MODEL=gemini-2.0-flash` en `.env` |
| `No existe la imagen` | Pasa una ruta válida o usa `--imagen` sin args |
