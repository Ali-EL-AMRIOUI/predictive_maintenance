import pandas as pd
import os

def clean_and_save_data(input_path, output_path):
    # 1. Définition des noms de colonnes (car le fichier brut n'en a pas)
    index_names = ['unit', 'cycle']
    setting_names = ['os1', 'os2', 'os3']
    sensor_names = [f's{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    # 2. Lecture du fichier brut
    print(f" Lecture de : {input_path}")
    df = pd.read_csv(input_path, sep='\s+', header=None, names=col_names)
    
    # 3. Calcul de la RUL (Remaining Useful Life) - Ta cible mathématique
    # Pour chaque moteur (unit), on trouve son cycle maximum
    max_cycle = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = max_cycle - df['cycle']
    
    # 4. Sauvegarde dans le dossier processed
    df.to_csv(output_path, index=False)
    print(f" Fichier nettoyé sauvegardé dans : {output_path}")

if __name__ == "__main__":
    # Chemins relatifs à la racine de ton projet
    RAW_PATH = "data/raw/train_FD001.txt"
    PROCESSED_PATH = "data/processed/train_cleaned.csv"
    
    if os.path.exists(RAW_PATH):
        clean_and_save_data(RAW_PATH, PROCESSED_PATH)
    else:
        print(f"❌ Erreur : Le fichier {RAW_PATH} est introuvable. Télécharge-le d'abord !")