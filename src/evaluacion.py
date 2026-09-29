"""Evaluación de los resultados de Gemini frente al ground truth manual.

Texto:
- accuracy de detección de idioma
- precision / recall / F1 de entidades (global y por tipo), con matching difuso
- accuracy de polaridad + matriz de confusión
- ROUGE-L aproximado del resumen (LCS de tokens frente al texto) y cobertura de palabras clave

Imágenes:
- OCR: Character Error Rate (distancia de Levenshtein / longitud de referencia)
- Descripción: precision / recall de elementos clave (con sinónimos)
- Escena: accuracy frente a la etiqueta manual (con sinónimos)
- Objetos: presencia/ausencia de cada objeto esperado + error de conteo

Uso:
    python -m src.evaluacion              # evalúa texto e imágenes
    python -m src.evaluacion --solo texto
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from src.utils import DATA_DIR, RESULTS_DIR, guardar_json

TIPOS = ["PERSONA", "ORGANIZACION", "UBICACION", "FECHA", "OTRO"]
POLARIDADES = ["positivo", "neutro", "negativo"]


# ----------------------------------------------------------------- helpers
def normalizar(s: str) -> str:
    """Minúsculas, sin tildes ni puntuación, espacios colapsados."""
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s$€%]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> list[str]:
    """Tokeniza un texto normalizado."""
    return normalizar(s).split()


def coincide(a: str, b: str) -> bool:
    """Matching difuso de menciones: igualdad normalizada o inclusión de una en otra."""
    na, nb = normalizar(a), normalizar(b)
    if not na or not nb:
        return False
    return na == nb or (len(min(na, nb, key=len)) >= 3 and (na in nb or nb in na))


def levenshtein(a: str, b: str) -> int:
    """Distancia de edición a nivel de carácter (programación dinámica, O(n·m))."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lcs(a: list[str], b: list[str]) -> int:
    """Longitud de la subsecuencia común más larga (base de ROUGE-L)."""
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall y F1 a partir de conteos."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


# ----------------------------------------------------------------- texto
def emparejar_entidades(pred: list[dict], gold: list[dict], usar_tipo: bool = True):
    """Empareja entidades predichas con las de referencia (greedy, 1 a 1).

    Returns:
        (pares, predichas_sin_pareja, gold_sin_pareja)
    """
    libres = list(gold)
    pares, fp = [], []
    for p in pred:
        idx = next((i for i, g in enumerate(libres)
                    if coincide(p["texto"], g["texto"]) and (not usar_tipo or p["tipo"] == g["tipo"])), None)
        if idx is None:
            fp.append(p)
        else:
            pares.append((p, libres.pop(idx)))
    return pares, fp, libres


def evaluar_texto(resultados: list[dict], gt: list[dict]) -> dict:
    """Calcula todas las métricas de la Parte 1 y el detalle por texto."""
    gt_by_id = {g["id"]: g for g in gt}
    por_tipo = {t: [0, 0, 0] for t in TIPOS}  # tp, fp, fn
    tot_sin_tipo = [0, 0, 0]
    confusion = {g: {p: 0 for p in POLARIDADES} for g in POLARIDADES}
    detalle = []
    ok_idioma = ok_sent = n = 0
    rouge, cobertura = [], []

    for r in resultados:
        if "error" in r or r["id"] not in gt_by_id:
            continue
        g = gt_by_id[r["id"]]
        n += 1
        idioma_ok = r["idioma"] == g["idioma"]
        ok_idioma += idioma_ok
        pol = r["sentimiento"]["polaridad"]
        sent_ok = pol == g["sentimiento"]
        ok_sent += sent_ok
        if pol in POLARIDADES:
            confusion[g["sentimiento"]][pol] += 1

        pares, fp, fn = emparejar_entidades(r["entidades"], g["entidades"], usar_tipo=True)
        for p, _ in pares:
            por_tipo[p["tipo"]][0] += 1
        for p in fp:
            por_tipo.setdefault(p["tipo"], [0, 0, 0])[1] += 1
        for x in fn:
            por_tipo[x["tipo"]][2] += 1
        p2, fp2, fn2 = emparejar_entidades(r["entidades"], g["entidades"], usar_tipo=False)
        tot_sin_tipo[0] += len(p2); tot_sin_tipo[1] += len(fp2); tot_sin_tipo[2] += len(fn2)

        # ROUGE-L aproximado: LCS resumen vs texto original (sin resumen de referencia)
        rs, tx = tokens(r.get("resumen", "")), tokens(r["texto"])
        l = lcs(rs, tx)
        pr = l / len(rs) if rs else 0.0
        rc = l / len(tx) if tx else 0.0
        rl = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        rouge.append(rl)
        claves = g.get("palabras_clave", [])
        res_norm = normalizar(r.get("resumen", ""))
        cub = sum(normalizar(k) in res_norm for k in claves) / len(claves) if claves else 0.0
        cobertura.append(cub)

        ent_f1 = prf(len(pares), len(fp), len(fn))["f1"]
        detalle.append({
            "id": r["id"], "tipo": r.get("tipo"), "subtipo": r.get("subtipo"),
            "idioma_pred": r["idioma"], "idioma_gt": g["idioma"], "idioma_ok": idioma_ok,
            "sent_pred": pol, "sent_gt": g["sentimiento"], "sent_ok": sent_ok,
            "confianza": r["sentimiento"]["confianza"], "entidades_f1": ent_f1,
            "rouge_l": round(rl, 3), "cobertura_clave": round(cub, 3),
            "entidades_fp": [e["texto"] for e in fp], "entidades_fn": [e["texto"] for e in fn],
        })

    tp = sum(v[0] for v in por_tipo.values()); fp_ = sum(v[1] for v in por_tipo.values())
    fn_ = sum(v[2] for v in por_tipo.values())
    return {
        "n_textos": n,
        "idioma_accuracy": round(ok_idioma / n, 3) if n else 0,
        "sentimiento_accuracy": round(ok_sent / n, 3) if n else 0,
        "entidades_micro": prf(tp, fp_, fn_),
        "entidades_sin_tipo": prf(*tot_sin_tipo),
        "entidades_por_tipo": {t: prf(*v) for t, v in por_tipo.items()},
        "rouge_l_medio": round(sum(rouge) / len(rouge), 3) if rouge else 0,
        "cobertura_clave_media": round(sum(cobertura) / len(cobertura), 3) if cobertura else 0,
        "matriz_confusion_sentimiento": confusion,
        "detalle": detalle,
    }


# ----------------------------------------------------------------- imágenes
def cer(pred: str, ref: str) -> float:
    """Character Error Rate con espacios/saltos colapsados (tope 1.0 salvo inserciones masivas)."""
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip()
    p, r = norm(pred), norm(ref)
    if not r:
        return 0.0 if not p else 1.0
    return levenshtein(p, r) / len(r)


def en_sinonimos(pred: str, clave: str, sinonimos: dict) -> bool:
    """¿La predicción corresponde a ``clave`` (o a alguno de sus sinónimos)?"""
    candidatos = [clave] + sinonimos.get(clave, [])
    return any(coincide(pred, c) for c in candidatos)


def evaluar_imagenes(resultados: list[dict], gt: dict) -> dict:
    """Calcula las métricas de la Parte 2 por imagen y agregadas."""
    sin = gt.get("_sinonimos", {})
    detalle = []
    for r in resultados:
        g = gt.get(r["imagen"])
        if not g:
            continue
        d = {"imagen": r["imagen"]}

        ocr = r.get("ocr", {})
        if "texto" in ocr:
            d["cer"] = round(cer(ocr["texto"], g["texto_ocr"]), 3)
            d["tiene_texto"] = bool(g["texto_ocr"])

        desc = r.get("descripcion", {})
        if "elementos_clave" in desc:
            pred = desc["elementos_clave"]
            ref = g["elementos_clave"]
            texto_desc = desc.get("descripcion", "") + " " + " ".join(pred)
            recall = sum(any(coincide(p, k) for p in pred) or normalizar(k) in normalizar(texto_desc)
                         for k in ref) / len(ref)
            # precision: elementos predichos que aparecen en la referencia o sus sinónimos
            universo = ref + [s for k in g.get("objetos", {}) for s in [k] + sin.get(k, [])]
            precision = sum(any(coincide(p, u) for u in universo) for p in pred) / len(pred) if pred else 0.0
            d["desc_recall"] = round(recall, 3)
            d["desc_precision"] = round(precision, 3)

        esc = r.get("escena", {})
        if "escena" in esc:
            d["escena_pred"] = esc["escena"]
            d["escena_gt"] = g["escena"]
            d["escena_ok"] = en_sinonimos(esc["escena"], g["escena"], sin)
            d["escena_confianza"] = esc.get("confianza")

        obj = r.get("objetos", {})
        if "objetos" in obj:
            pred = obj["objetos"]
            presentes, err_conteo = 0, []
            for nombre, cant in g["objetos"].items():
                match = [o for o in pred if en_sinonimos(o["nombre"], nombre, sin)]
                if match:
                    presentes += 1
                    err_conteo.append(abs(sum(int(o.get("cantidad", 0)) for o in match) - cant) / cant)
            d["objetos_presencia"] = round(presentes / len(g["objetos"]), 3)
            d["objetos_error_conteo"] = round(sum(err_conteo) / len(err_conteo), 3) if err_conteo else None
            d["objetos_detectados"] = {o["nombre"]: o.get("cantidad") for o in pred}
        detalle.append(d)

    media = lambda k, filtro=lambda d: True: (
        round(sum(d[k] for d in detalle if k in d and d[k] is not None and filtro(d))
              / max(1, sum(1 for d in detalle if k in d and d[k] is not None and filtro(d))), 3))
    return {
        "n_imagenes": len(detalle),
        "cer_medio_con_texto": media("cer", lambda d: d.get("tiene_texto")),
        "desc_recall_medio": media("desc_recall"),
        "desc_precision_media": media("desc_precision"),
        "escena_accuracy": round(sum(d.get("escena_ok", False) for d in detalle) / max(1, len(detalle)), 3),
        "objetos_presencia_media": media("objetos_presencia"),
        "objetos_error_conteo_medio": media("objetos_error_conteo"),
        "detalle": detalle,
    }


def main() -> int:
    """CLI: evalúa y guarda resultados/metricas.json."""
    p = argparse.ArgumentParser(description="Evaluación frente a ground truth")
    p.add_argument("--solo", choices=["texto", "imagenes"])
    p.add_argument("--resultados", default=str(RESULTS_DIR))
    p.add_argument("--salida", default=str(RESULTS_DIR / "metricas.json"))
    args = p.parse_args()
    res_dir = Path(args.resultados)
    metricas = {}

    if args.solo in (None, "texto"):
        f = res_dir / "resultados_texto.json"
        if f.exists():
            gt = json.loads((DATA_DIR / "ground_truth_texto.json").read_text(encoding="utf-8"))
            metricas["texto"] = evaluar_texto(json.loads(f.read_text(encoding="utf-8")), gt)
            m = metricas["texto"]
            print(f"TEXTO   idioma={m['idioma_accuracy']:.0%}  sentimiento={m['sentimiento_accuracy']:.0%}  "
                  f"entidades F1={m['entidades_micro']['f1']:.2f}  ROUGE-L≈{m['rouge_l_medio']:.2f}  "
                  f"cobertura={m['cobertura_clave_media']:.0%}")
        else:
            print(f"(sin {f}; ejecuta src.text_analysis)")

    if args.solo in (None, "imagenes"):
        f = res_dir / "resultados_imagenes.json"
        if f.exists():
            gt = json.loads((DATA_DIR / "ground_truth_imagenes.json").read_text(encoding="utf-8"))
            metricas["imagenes"] = evaluar_imagenes(json.loads(f.read_text(encoding="utf-8")), gt)
            m = metricas["imagenes"]
            print(f"IMAGEN  CER={m['cer_medio_con_texto']:.3f}  desc R={m['desc_recall_medio']:.2f} "
                  f"P={m['desc_precision_media']:.2f}  escena={m['escena_accuracy']:.0%}  "
                  f"objetos={m['objetos_presencia_media']:.0%}")
        else:
            print(f"(sin {f}; ejecuta src.image_analysis)")

    if metricas:
        guardar_json(metricas, Path(args.salida))
        print(f"✓ Métricas en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
