import serial
import time
import numpy as np
from collections import deque
import re

# Stocker les 10 dernières valeurs de masse
historique_vwr = deque(maxlen=10)
historique_kern = deque(maxlen=10)
historique_temps = deque(maxlen=10)
filename = "1p3_1000ml_v2.txt"

# Fonction pour obtenir l'heure au format minutes:secondes:millisecondes


def obtenir_temps():
    t = time.time()
    minutes = int(t // 60)
    secondes = int(t % 60)
    millisecondes = int((t % 1) * 1000)  # Extraire les millisecondes
    # Retourne aussi le temps en secondes pour le fit
    return f"{minutes:02}:{secondes:02}:{millisecondes:03}", t

# Fonction pour ouvrir et lire les données d'une balance


def lire_balance(port):
    try:
        # Ouvrir le port série
        ser = serial.Serial(port, baudrate=9600, timeout=1)
        # Lire les données de la balance
        while True:
            data = ser.readline().decode().strip()
            result = re.search(r'([+-]?\d+\.\d+)', data)
            if data:
                if port == "COM4":
                    return float(result.group(0))/0.791
                else:
                    return float(result.group(0))/0.997
    except Exception as e:
        print(f"Erreur avec {port}: {e}")
        return None
    finally:
        ser.close()

# Fonction pour calculer le débit à partir des 10 dernières mesures


def calculer_debit(historique_masse, historique_temps):
    if len(historique_masse) < 2:
        return 0  # Pas assez de points pour un fit

    # Ajustement linéaire (y = ax + b) -> on récupère a (pente)
    t_array = np.array(historique_temps)
    m_array = np.array(historique_masse)

    # Remise à zéro en soustrayant la première valeur stockée
    t_array -= t_array[0]  # Temps relatif par rapport au premier
    m_array -= m_array[0]  # Masse relative par rapport à la première mesure

    # Fit linéaire d'ordre 1 (affine y = ax + b)
    pente, _ = np.polyfit(t_array, -m_array, 1)
    return pente  # La pente représente le débit

# Fonction principale pour collecter et enregistrer les données


def collecter_donnees():
    with open(filename, "w") as fichier:
        while True:
            # Lire les données des balances
            masse_vwr = lire_balance("COM3")
            masse_kern = lire_balance("COM4")
            temps_format, temps_seconde = obtenir_temps()

            if masse_vwr is not None and masse_kern is not None:
                # Stocker les valeurs dans l'historique
                historique_vwr.append(masse_vwr)
                historique_kern.append(masse_kern)
                historique_temps.append(temps_seconde)

                # Calculer le débit (mais ne pas l'enregistrer dans le fichier)
                debit_vwr = calculer_debit(historique_vwr, historique_temps)*60
                debit_kern = calculer_debit(
                    historique_kern, historique_temps)*60
                total_flow = debit_vwr+debit_kern
                ratio_flow = debit_vwr / debit_kern if debit_kern != 0 else 0

                # Enregistrer les données dans le fichier (sans les valeurs de débit)
                fichier.write(
                    f"{masse_vwr} {temps_format} {masse_kern} {temps_format}\n")

                # Afficher les valeurs et le débit à l'écran
                # Efface la console pour afficher proprement
                print("\033[H\033[J")
                print("==============================")
                print(f"Volume EtOH  : {masse_kern:.2f} ml")
                print(f"Volume H2O  : {masse_vwr:.2f} mL")
                print(f"Débit Kern  : {debit_kern:.2f} mL/min")
                print(f"Débit VWR   : {debit_vwr:.2f} mL/min")
                print(f"Somme débits: {total_flow:.2f} mL/min")
                print(f"Ratio VWR/Kern: {ratio_flow:.2f}")
                print("==============================")
            # Attendre 0.1s avant la prochaine mesure
            time.sleep(0.1)


# Appel de la fonction principale
if __name__ == "__main__":
    collecter_donnees()
