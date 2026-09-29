# Incidencia: acceso a la API de Gemini (403 / 404 / 401)

**Fecha:** 29-09-2026
**Estado:** sin resolver. Es un bloqueo externo, a nivel de proyecto y cuenta de Google, que no se puede anular desde el código.
**Impacto:** no se pudieron obtener resultados reales. El código, los datos, el *ground truth*, la evaluación y el notebook están terminados y verificados con un cliente simulado. Basta con una clave válida para ejecutar todo.

## 1. Cronología de lo observado

| # | Clave | Modelo | Respuesta de la API |
|---|---|---|---|
| 1 | Clave 1 (`AQ.…`) | `gemini-2.5-flash` | **404 NOT_FOUND**: *"This model models/gemini-2.5-flash is no longer available to new users."* |
| 2 | Clave 1 | `gemini-3.8-flash`, `3.7-flash`, `3.6-flash`, `3.5-flash`, `flash-latest`, `3-flash-preview`, `3.5-flash-lite` | **403 PERMISSION_DENIED**: *"Your project has been denied access. Please contact support."* |
| 3 | Clave 1 | `gemini-2.5-flash-lite`, `gemini-2.5-pro` | **404**: modelo retirado para cuentas nuevas |
| 4 | Clave 2 (`AQ.…`) | Flash, Pro y Gemma | Igual: **404** en 2.5 y **403 "project denied access"** en el resto |
| 5 | Clave 3 (`AQ.…`, generada a partir de otra) | `gemini-3.8-flash`, `flash-latest`, `3.1-flash-lite` | **200 OK** en una prueba de "Hola" |
| 6 | Clave 3 | `gemini-3.8-flash` (ejecución real) | **503 UNAVAILABLE** (*"high demand"*) en los textos 1-2 y, desde el texto 3, **401 UNAUTHENTICATED** con motivo `ACCESS_TOKEN_TYPE_UNSUPPORTED`. Después, ni `models.list()` funciona. |

Con las claves 1 y 2, `models.list()` **sí** funcionaba. La autenticación era correcta, pero el proyecto no tenía permiso para generar contenido.

## 2. Diagnóstico

1. **Modelo retirado (404):** `gemini-2.5-flash`, el modelo que pide el enunciado, ya no se ofrece a cuentas nuevas. Se resuelve con `GEMINI_MODEL` en `.env`, sin tocar código.
2. **Proyecto bloqueado (403 "Your project has been denied access"):** no es un error de clave, de restricciones ni de API deshabilitada, porque esos casos dan mensajes distintos (`API_KEY_INVALID`, `SERVICE_DISABLED`, "referer"…). Google ha bloqueado el proyecto o la cuenta. Solo lo puede levantar Google (soporte o revisión de la cuenta).
3. **Clave invalidada (401):** la clave 3 dejó de ser válida a los pocos minutos de funcionar. Probablemente se revocó por regenerarla o borrarla, o por la detección automática de claves filtradas (las claves se compartieron por chat).

## 3. Qué se ha hecho en el código por esta incidencia

- `GEMINI_MODEL` se configura en `.env`. El valor por defecto sigue siendo `gemini-2.5-flash`, como pide el enunciado.
- `src/utils.py` distingue los errores:
  - **No reintentables:** `API_KEY_INVALID`, `PERMISSION_DENIED`, `UNAUTHENTICATED` / 401. Paran el lote al instante con un mensaje claro, sin gastar cuota.
  - **Reintentables:** 429, 500 y 503. Se reintenta esperando cada vez más (2, 4, 8 y 16 s).
- Todo el flujo (análisis de texto e imagen, evaluación y notebook) se verificó con un cliente simulado. El simulado provoca a propósito un 429 y un JSON roto para comprobar el reintento y el prompt de corrección.

## 4. Cómo reanudar cuando haya una clave válida

1. En https://aistudio.google.com/apikey: **Create API key → Create in new project** (con otra cuenta si el bloqueo es de la cuenta).
2. Comprueba que funciona:
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); from google import genai; print(genai.Client().models.generate_content(model='gemini-flash-latest', contents='Hola').text)"
   ```
3. En `.env`, pon `GEMINI_API_KEY=...` y `GEMINI_MODEL=gemini-flash-latest` (o cualquier modelo que responda).
4. Ejecuta:
   ```bash
   python -m src.text_analysis && python -m src.image_analysis && python -m src.evaluacion
   jupyter nbconvert --to notebook --execute --inplace notebooks/analisis_resultados.ipynb
   ```
   Tiempo aproximado: 3-4 min (40 llamadas de texto y 24 de imagen, con 1 s de pausa).
