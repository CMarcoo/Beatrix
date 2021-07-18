import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import ffmpeg
import sklearn


class Regione:
    def __init__(self, nome, id):
        self.nome = nome
        self.id = id


class Zona:
    def __init__(self, regioni):
        self.regioni = regioni


class Stato:
    def __init__(self, zone):
        self.zone = zone


mappa_cartelle = {1: "aosta",
                  2: "trentino",
                  3: "piemonte",
                  4: "friuli",
                  5: "lombardia",
                  6: "liguria",
                  7: "emilia",
                  8: "veneto",
                  9: "lazio",
                  10: "umbria",
                  11: "marche",
                  12: "toscana",
                  13: "molise",
                  14: "campania",
                  15: "basilicata",
                  16: "abruzzo",
                  17: "puglia",
                  18: "calabria",
                  19: "sicilia",
                  20: "sardegna"}

# ----------------------------------------
# NORD

aosta = Regione("Aosta", 1)
trentino = Regione("Tretino", 2)
piemonte = Regione("Piemonte", 3)
friuli = Regione("Friuli-Venezia-Giulia", 4)

zona_nord = Zona([aosta, trentino, piemonte, friuli])

# ----------------------------------------
# CENTRO-NORD

lombardia = Regione("lombardia", 5)
liguria = Regione("Liguria", 6)
emilia = Regione("Emilia-Romagna", 7)
veneto = Regione("Veneto", 8)

zona_centro_nord = Zona([liguria, emilia, lombardia])

# ----------------------------------------
# CENTRO

lazio = Regione("Lazio", 9)
umbria = Regione("Umbria", 10)
marche = Regione("Marche", 11)
toscana = Regione("Toscana", 12)

zona_centro = Zona([toscana, umbria, marche, lazio])

# ----------------------------------------
# CENTRO SUD

molise = Regione("Molise", 13)
campania = Regione("Campania", 14)
basilicata = Regione("Basilicata", 15)
abruzzo = Regione("Abruzzo", 16)

zona_centro_sud = Zona([abruzzo, molise, campania, basilicata])

# ----------------------------------------
# SUD

puglia = Regione("Puglia", 17)
calabria = Regione("Calabria", 18)
sicilia = Regione("Sicilia", 19)
sardegna = Regione("Sardegna", 20)

zona_sud = Zona([sicilia, sardegna, puglia, calabria])

# ---------------------------------------

italia = Stato([zona_sud, zona_centro_sud, zona_centro_nord, zona_nord])

cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8, 8))
for zona in italia.zone:
    for regione in zona.regioni:
        # pathlib.Path(f'audio/{regione}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'./audio/{mappa_cartelle[regione.id]}'):
            songname = f'./audio/{mappa_cartelle[regione.id]}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=5)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
            plt.axis('off')
            plt.savefig(f'img_data/{zona}/{filename[:-3].replace(".", "")}.png')
            plt.clf()
