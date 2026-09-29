"""Genera las 5 imágenes sintéticas de prueba (reproducibles) en data/imagenes/.

Se pueden sustituir por fotos reales; en ese caso actualiza data/ground_truth_imagenes.json.
"""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent.parent / "data" / "imagenes"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(42)

def font(size, bold=False):
    for p in (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else []) + [
        "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try: return ImageFont.truetype(p, size)
        except OSError: pass
    return ImageFont.load_default()

DOC_TEXT = """INFORME TRIMESTRAL DE VENTAS
Empresa: Distribuciones Norte S.L.
Fecha: 30 de junio de 2026

1. Resumen
Las ventas del segundo trimestre alcanzaron 1.245.000 euros,
un 12% más que en el mismo periodo del año anterior.

2. Detalle por región
Madrid: 540.000 euros
Barcelona: 410.000 euros
Valencia: 295.000 euros

3. Conclusión
Se recomienda ampliar la plantilla comercial en Valencia."""

def doc_escaneado():
    img = Image.new("RGB", (1000, 1300), (246, 243, 235))
    d = ImageDraw.Draw(img)
    y = 80
    for i, line in enumerate(DOC_TEXT.split("\n")):
        d.text((90, y), line, fill=(25, 25, 30), font=font(34 if i == 0 else 27, bold=(i == 0)))
        y += 58 if i == 0 else 44
    # ruido de escaneo + ligera rotación
    px = img.load()
    for _ in range(25000):
        x, yy = random.randrange(1000), random.randrange(1300)
        g = random.randint(170, 230); px[x, yy] = (g, g, g - 5)
    img = img.rotate(1.2, expand=False, fillcolor=(230, 228, 220), resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img.save(OUT / "doc_escaneado.png")

def coche(d, x, y, color):
    d.rounded_rectangle([x, y, x + 170, y + 55], 12, fill=color)
    d.polygon([(x + 35, y), (x + 55, y - 38), (x + 120, y - 38), (x + 145, y)], fill=color)
    d.polygon([(x + 50, y - 4), (x + 63, y - 32), (x + 115, y - 32), (x + 132, y - 4)], fill=(170, 205, 230))
    for cx in (x + 38, x + 132):
        d.ellipse([cx - 20, y + 38, cx + 20, y + 78], fill=(20, 20, 20))
        d.ellipse([cx - 8, y + 50, cx + 8, y + 66], fill=(150, 150, 150))

def persona(d, x, y, color):
    d.ellipse([x - 12, y, x + 12, y + 24], fill=(225, 185, 150))
    d.rounded_rectangle([x - 16, y + 26, x + 16, y + 85], 8, fill=color)
    d.rectangle([x - 13, y + 85, x - 3, y + 130], fill=(40, 40, 70))
    d.rectangle([x + 3, y + 85, x + 13, y + 130], fill=(40, 40, 70))

def foto_urbana():
    W, H = 1280, 800
    img = Image.new("RGB", (W, H), (150, 195, 235)); d = ImageDraw.Draw(img)
    x = 0
    for c, h in [((180,120,90),420),((210,200,180),520),((120,130,150),600),((200,160,110),460),((160,90,80),550),((190,190,200),480)]:
        w = 215; d.rectangle([x, H-200-h, x+w, H-200], fill=c)
        for wy in range(H-200-h+30, H-230, 60):
            for wx in range(x+22, x+w-30, 48):
                d.rectangle([wx, wy, wx+26, wy+36], fill=(235, 240, 200) if random.random()<.4 else (60,80,110))
        x += w
    d.rectangle([0, H-200, W, H-160], fill=(170,170,170))  # acera
    d.rectangle([0, H-160, W, H], fill=(60,60,65))        # calzada
    for lx in range(20, W, 120): d.rectangle([lx, H-85, lx+60, H-77], fill=(240,240,240))
    for px_, c in [(150,(200,40,40)),(420,(40,90,200)),(760,(230,190,40)),(1060,(60,160,80))]: persona(d, px_, H-330, c)
    coche(d, 90, H-120, (200,30,30)); coche(d, 520, H-120, (30,30,160)); coche(d, 950, H-120, (230,230,230))
    # semáforo
    d.rectangle([640, H-420, 652, H-200], fill=(50,50,50)); d.rounded_rectangle([628, H-470, 664, H-380], 6, fill=(30,30,30))
    for i,c in enumerate([(220,0,0),(80,60,0),(0,70,0)]): d.ellipse([636, H-464+i*28, 656, H-444+i*28], fill=c)
    img.save(OUT / "foto_urbana.jpg", quality=92)

def producto():
    W, H = 1000, 1000
    img = Image.new("RGB", (W, H), (250, 250, 250)); d = ImageDraw.Draw(img)
    d.ellipse([290, 780, 710, 840], fill=(225, 225, 225))                 # sombra
    d.rounded_rectangle([330, 360, 670, 810], 40, fill=(200, 45, 45))     # taza
    d.ellipse([330, 330, 670, 400], fill=(170, 30, 30)); d.ellipse([350, 340, 650, 392], fill=(80, 45, 25))  # café
    d.ellipse([630, 450, 800, 680], outline=(200,45,45), width=38)        # asa
    d.text((420, 560), "CAFÉ", fill=(255,255,255), font=font(64, bold=True))
    img.save(OUT / "producto.jpg", quality=92)

def paisaje():
    W, H = 1280, 760
    img = Image.new("RGB", (W, H)); d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H; d.line([(0,y),(W,y)], fill=(int(120+100*t), int(170+60*t), int(235-20*t)))
    d.ellipse([1020, 90, 1120, 190], fill=(255, 230, 120))
    for pts, c in [([(0,560),(300,220),(560,560)],(110,120,140)),([(380,560),(760,150),(1140,560)],(90,100,125)),([(900,560),(1150,280),(1280,400),(1280,560)],(120,130,150))]:
        d.polygon(pts, fill=c)
    d.polygon([(760,150),(700,240),(735,225),(760,250),(790,228),(820,240)], fill=(250,250,255))
    d.polygon([(300,220),(255,285),(300,270),(340,280)], fill=(250,250,255))
    d.rectangle([0,560,W,H], fill=(80,140,70))
    d.ellipse([-100, 620, 1400, 900], fill=(60,110,160))  # lago
    for x in range(30, W, 70):
        h = random.randint(70, 120); d.rectangle([x+12, 600-10, x+18, 600+15], fill=(90,60,30))
        d.polygon([(x, 595), (x+15, 595-h), (x+30, 595)], fill=(30, 90, 45))
    img.save(OUT / "paisaje.jpg", quality=92)

def screenshot():
    W, H = 1280, 800
    img = Image.new("RGB", (W, H), (245, 246, 248)); d = ImageDraw.Draw(img)
    d.rectangle([0,0,W,40], fill=(220,222,226))
    for i,c in enumerate([(255,95,87),(254,188,46),(40,200,64)]): d.ellipse([15+i*24,12,31+i*24,28], fill=c)
    d.rounded_rectangle([140,8,900,32], 8, fill=(255,255,255)); d.text((155,11), "https://tienda-ejemplo.com/ofertas", fill=(60,60,60), font=font(17))
    d.rectangle([0,40,W,110], fill=(30,60,120)); d.text((40,58), "TiendaEjemplo", fill=(255,255,255), font=font(34, bold=True))
    for i,t in enumerate(["Inicio","Ofertas","Mi cuenta"]): d.text((820+i*140,68), t, fill=(220,230,255), font=font(22))
    d.text((60,150), "Oferta de otoño: 30% de descuento", fill=(20,20,20), font=font(44, bold=True))
    d.text((60,215), "Válido hasta el 15 de octubre. Envío gratis en pedidos superiores a 50 €.", fill=(70,70,70), font=font(24))
    d.rounded_rectangle([60,280,360,340], 10, fill=(230,90,30)); d.text((105,293), "Comprar ahora", fill=(255,255,255), font=font(28, bold=True))
    # notificación superpuesta
    d.rounded_rectangle([820,600,1240,760], 14, fill=(40,40,45))
    d.text((845,620), "Nuevo mensaje", fill=(255,255,255), font=font(24, bold=True))
    d.text((845,660), "Tu pedido #48213 ha sido enviado", fill=(210,210,210), font=font(20))
    d.text((845,700), "hace 2 minutos", fill=(150,150,150), font=font(18))
    img.save(OUT / "screenshot.png")

if __name__ == "__main__":
    doc_escaneado(); foto_urbana(); producto(); paisaje(); screenshot()
    print("Imágenes generadas en", OUT)
