"""Genera data/textos.jsonl y data/ground_truth_texto.json.

ids 1-10: los 10 textos originales de main.py (TEXTOS_ES + TEXTOS_EN), leídos del propio fichero.
ids 11-20: ampliación para cubrir las categorías del reto (noticias ES/EN/PT, emails formales,
reseña mixta, sarcasmo e idioma mixto), que los textos originales no incluyen.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def textos_main():
    """Extrae TEXTOS_ES y TEXTOS_EN de main.py sin importarlo (evita crear el cliente)."""
    arbol = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    val = {}
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and isinstance(nodo.targets[0], ast.Name):
            if nodo.targets[0].id in ("TEXTOS_ES", "TEXTOS_EN"):
                val[nodo.targets[0].id] = ast.literal_eval(nodo.value)
    return val["TEXTOS_ES"], val["TEXTOS_EN"]


def E(texto, tipo):
    return {"texto": texto, "tipo": tipo}


ES, EN = textos_main()
AMPLIACION = [
    "El Gobierno de España aprobó ayer en Madrid un plan de 2.000 millones de euros para modernizar la red ferroviaria, según anunció el ministro Óscar Puente.",
    "Apple CEO Tim Cook announced on Monday in Cupertino that the company will invest $5 billion in AI research over the next three years.",
    "O presidente Lula anunciou em Brasília um novo programa de habitação para 2 milhões de famílias, em parceria com a Caixa Econômica Federal.",
    "Compré los auriculares SoundMax Pro y son una maravilla: el sonido es nítido, la batería dura días y la cancelación de ruido funciona de lujo.",
    "La batidora Kitchenix llegó rota, el servicio técnico no contesta y me han cobrado dos veces. No la recomiendo a nadie.",
    "The Nimbus 3 laptop has a gorgeous screen and a great keyboard, but the battery barely lasts four hours and the fans are loud.",
    "Estimada Sra. López: le confirmamos que la reunión con el equipo de Iberdrola tendrá lugar el 15 de octubre de 2026 en nuestras oficinas de Bilbao. Atentamente, Carlos Ruiz.",
    "Dear Mr. Thompson, following our call on March 3rd, please find attached the revised contract from Deloitte. Kind regards, Emma Watson, Legal Department.",
    "Genial, otra vez el tren de Renfe con dos horas de retraso. Qué maravilla de servicio, de verdad, justo lo que necesitaba hoy.",
    "Ayer tuve el meeting con el team de marketing de Google y honestly the feedback was great, aunque el deadline del viernes es imposible.",
]

# (texto, origen, tipo, subtipo, idioma, sentimiento, entidades, palabras_clave, nota)
FILAS = [
    (ES[0], "28-09-Reto", "resena", "positiva", "es", "positivo", [E("Rosalía", "PERSONA")], ["Rosalía", "disco", "obra maestra"], ""),
    (ES[1], "28-09-Reto", "resena", "negativa", "es", "negativo", [], ["servicio", "compañía", "tres semanas"], ""),
    (ES[2], "28-09-Reto", "resena", "mixta", "es", "neutro", [E("calle Mayor 12", "UBICACION")], ["restaurante", "calle Mayor", "correcta"], "Valoración tibia ('correcta, sin más') → neutro."),
    (ES[3], "28-09-Reto", "personal", "positiva", "es", "positivo", [E("Madrid", "UBICACION")], ["familia", "Madrid", "dos años"], ""),
    (ES[4], "28-09-Reto", "resena", "negativa", "es", "negativo", [E("Iberia", "ORGANIZACION")], ["vuelo", "Iberia", "5 horas", "conexión"], ""),
    (EN[0], "28-09-Reto", "resena", "positiva", "en", "positivo", [E("iPhone", "OTRO"), E("Paris", "UBICACION")], ["iPhone", "camera", "Paris"], ""),
    (EN[1], "28-09-Reto", "resena", "negativa", "en", "negativo", [], ["movie", "plot", "acting"], ""),
    (EN[2], "28-09-Reto", "email", "en", "en", "neutro", [E("Monday at 10am", "FECHA"), E("London", "UBICACION")], ["meeting", "Monday", "London"], ""),
    (EN[3], "28-09-Reto", "personal", "positiva", "en", "positivo", [E("Google", "ORGANIZACION")], ["job", "Google", "team"], ""),
    (EN[4], "28-09-Reto", "resena", "negativa", "en", "negativo", [E("Amazon", "ORGANIZACION")], ["Amazon", "broken", "customer support"], ""),
    (AMPLIACION[0], "ampliacion", "noticia", "es", "es", "neutro",
     [E("Gobierno de España", "ORGANIZACION"), E("Madrid", "UBICACION"), E("ayer", "FECHA"), E("Óscar Puente", "PERSONA"), E("2.000 millones de euros", "OTRO")],
     ["Gobierno", "2.000 millones", "red ferroviaria", "Óscar Puente"], ""),
    (AMPLIACION[1], "ampliacion", "noticia", "en", "en", "neutro",
     [E("Apple", "ORGANIZACION"), E("Tim Cook", "PERSONA"), E("Monday", "FECHA"), E("Cupertino", "UBICACION"), E("$5 billion", "OTRO")],
     ["Apple", "Tim Cook", "5 billion", "AI"], ""),
    (AMPLIACION[2], "ampliacion", "noticia", "pt", "pt", "neutro",
     [E("Lula", "PERSONA"), E("Brasília", "UBICACION"), E("Caixa Econômica Federal", "ORGANIZACION")],
     ["Lula", "habitação", "2 milhões", "Caixa"], ""),
    (AMPLIACION[3], "ampliacion", "resena", "positiva", "es", "positivo", [E("SoundMax Pro", "OTRO")],
     ["auriculares", "SoundMax", "sonido", "batería"], ""),
    (AMPLIACION[4], "ampliacion", "resena", "negativa", "es", "negativo", [E("Kitchenix", "OTRO")],
     ["batidora", "rota", "servicio técnico", "dos veces"], "Kitchenix = marca de producto → OTRO."),
    (AMPLIACION[5], "ampliacion", "resena", "mixta", "en", "neutro", [E("Nimbus 3", "OTRO")],
     ["Nimbus", "screen", "keyboard", "battery"], "Reseña mixta: el esquema no tiene clase 'mixto' → neutro."),
    (AMPLIACION[6], "ampliacion", "email", "es", "es", "neutro",
     [E("Sra. López", "PERSONA"), E("Iberdrola", "ORGANIZACION"), E("15 de octubre de 2026", "FECHA"), E("Bilbao", "UBICACION"), E("Carlos Ruiz", "PERSONA")],
     ["reunión", "Iberdrola", "15 de octubre", "Bilbao"], ""),
    (AMPLIACION[7], "ampliacion", "email", "en", "en", "neutro",
     [E("Mr. Thompson", "PERSONA"), E("March 3rd", "FECHA"), E("Deloitte", "ORGANIZACION"), E("Emma Watson", "PERSONA"), E("Legal Department", "ORGANIZACION")],
     ["contract", "Deloitte", "Thompson"], ""),
    (AMPLIACION[8], "ampliacion", "ambiguo", "sarcasmo", "es", "negativo", [E("Renfe", "ORGANIZACION"), E("hoy", "FECHA")],
     ["Renfe", "retraso", "dos horas"], "Sarcasmo: literal positivo, intención negativa."),
    (AMPLIACION[9], "ampliacion", "ambiguo", "idioma_mixto", "es", "positivo",
     [E("Google", "ORGANIZACION"), E("Ayer", "FECHA"), E("viernes", "FECHA")],
     ["meeting", "marketing", "Google", "feedback", "deadline"], "Code-switching ES/EN; idioma dominante 'es'."),
]

with open(ROOT / "data/textos.jsonl", "w", encoding="utf-8") as f:
    for i, (t, origen, tipo, sub, *_ ) in enumerate(FILAS, 1):
        f.write(json.dumps({"id": i, "origen": origen, "tipo": tipo, "subtipo": sub, "texto": t}, ensure_ascii=False) + "\n")
gt = [{"id": i, "idioma": idi, "sentimiento": s, "entidades": ents, "palabras_clave": pc, "nota": nota}
      for i, (_, _, _, _, idi, s, ents, pc, nota) in enumerate(FILAS, 1)]
(ROOT / "data/ground_truth_texto.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✓ {len(FILAS)} textos")
