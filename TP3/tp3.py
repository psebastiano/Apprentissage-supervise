import pandas as pd
import numpy as np
from random import uniform as rd
from typing import List, Dict, Any, Union
import matplotlib.pyplot as plt
import os
import copy
import dataframe_image as dfi
from itertools import product
from perceptron_visualizer import PerceptronVisualizer

seed = 42

def display_and_save_matrix_table(
    data_matrix: Union[List[List[float]], np.ndarray], 
    filename: str = "matrix_table.png", 
    title: str = "Tableau des données",
    index_name: str = "Numéro du Poids",
    col_name: str = "Col",  # Changement du nom par défaut pour la nouvelle structure
    batch_size: int = 100  # Nombre de lignes par batch
):
    # 1. Création du DataFrame
    # Note : Si 'data_matrix' est une seule liste (vecteur 1D), Pandas le traite comme une colonne.
    # On assure qu'il est bien un tableau 2D pour la transposition.
    if not isinstance(data_matrix[0], list) and not isinstance(data_matrix[0], np.ndarray):
        # Si c'est un vecteur 1D (comme une liste de 60 poids), on le met dans une liste de listes
        data_matrix = [data_matrix]

    df = pd.DataFrame(data_matrix)
    
    # 2. Transposition du DataFrame
    # Les colonnes (Poids_0, Poids_1, etc.) deviennent les lignes.
    # Les lignes (Essai/Ligne 0, 1, etc.) deviennent les colonnes.
    df = df.T 
    
    # 3. Renommer les colonnes restantes (qui sont les anciens indices de ligne/essais)
    # Ex: Poids du Perceptron 1, Poids du Perceptron 2, etc.
    df.columns = [f'{col_name} {i}' for i in df.columns]

    # 4. Ajout de l'index des poids comme colonne
    df.index.name = index_name
    df.reset_index(inplace=True)
    
    # 5. Calculer le nombre de batches nécessaires
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size  # Arrondi supérieur
    
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Nombre total de lignes : {total_rows}")
    print(f"Taille du batch : {batch_size}")
    print(f"Nombre de batches nécessaires : {num_batches}\n")
    
    all_dfs = []  # Pour stocker tous les DataFrames générés
    
    # 6. Traiter chaque batch séparément
    for batch_num in range(num_batches):
        start_row = batch_num * batch_size
        end_row = min((batch_num + 1) * batch_size, total_rows)
        
        # Créer un sous-DataFrame pour ce batch
        batch_df = df.iloc[start_row:end_row].copy()
        
        # Réinitialiser l'index pour ce batch
        batch_df.reset_index(drop=True, inplace=True)
        
        # Créer le titre du batch
        if num_batches == 1:
            batch_title = title
        else:
            batch_title = f"{title} (Batch {batch_num + 1}/{num_batches}) - Lignes {start_row + 1} à {end_row}"
        
        # 7. Application du style et du formatage pour ce batch
        styled_batch_df = batch_df.style.set_caption(batch_title).format(
            # Applique un formatage flottant à toutes les colonnes numériques
            {col: '{:.6f}' for col in batch_df.columns if col != index_name}
        ).set_properties(**{'text-align': 'center'})
        
        # 8. Affichage Console pour ce batch
        print(f"\n{'='*80}")
        print(f"BATCH {batch_num + 1}/{num_batches} - Lignes {start_row + 1} à {end_row}")
        print(f"{'='*80}\n")
        print(batch_df.to_string(index=False))
        
        # 9. Générer le nom de fichier pour ce batch
        if num_batches == 1:
            batch_filename = filename
        else:
            # Séparer l'extension et ajouter le suffixe batch
            name_without_ext, ext = os.path.splitext(filename)
            batch_filename = f"{name_without_ext}_batch_{batch_num + 1}{ext}"
        
        # 10. Sauvegarde en PNG pour ce batch
        try:
            # Utiliser max_rows=-1 et max_cols=-1 pour contourner la limitation
            dfi.export(styled_batch_df, batch_filename, max_rows=-1, max_cols=-1) 
            print(f"\n[OK] Batch {batch_num + 1} sauvegardé avec succès dans le fichier : **{batch_filename}**")
            
        except Exception as e:
            print(f"\n[ERR] Erreur lors de l'exportation du batch {batch_num + 1} en PNG : {e}")
        
        all_dfs.append(batch_df)
    
    # 11. Résumé final
    print(f"\n{'='*80}")
    print("RÉSUMÉ DE L'EXPORTATION")
    print(f"{'='*80}")
    print(f"Total de lignes traitées : {total_rows}")
    print(f"Taille du batch : {batch_size}")
    print(f"Nombre de batches générés : {num_batches}")
    
    if num_batches > 1:
        print(f"Fichiers générés :")
        for batch_num in range(num_batches):
            name_without_ext, ext = os.path.splitext(filename)
            batch_filename = f"{name_without_ext}_batch_{batch_num + 1}{ext}"
            print(f"  - {batch_filename}")
    
    # Retourner le DataFrame complet (non découpé) pour compatibilité
    return df

def display_and_save_matrix_table_old(
    data_matrix: Union[List[List[float]], np.ndarray], 
    filename: str = "matrix_table.png", 
    title: str = "Tableau des données",
    index_name: str = "Numéro du Poids",
    col_name: str = "Col" # Changement du nom par défaut pour la nouvelle structure
):
    # 1. Création du DataFrame
    # Note : Si 'data_matrix' est une seule liste (vecteur 1D), Pandas le traite comme une colonne.
    # On assure qu'il est bien un tableau 2D pour la transposition.
    if not isinstance(data_matrix[0], list) and not isinstance(data_matrix[0], np.ndarray):
        # Si c'est un vecteur 1D (comme une liste de 60 poids), on le met dans une liste de listes
        data_matrix = [data_matrix]

    df = pd.DataFrame(data_matrix)
    
    # 2. Transposition du DataFrame
    # Les colonnes (Poids_0, Poids_1, etc.) deviennent les lignes.
    # Les lignes (Essai/Ligne 0, 1, etc.) deviennent les colonnes.
    df = df.T 
    
    # 3. Renommer les colonnes restantes (qui sont les anciens indices de ligne/essais)
    # Ex: Poids du Perceptron 1, Poids du Perceptron 2, etc.
    df.columns = [f'{col_name} {i}' for i in df.columns]

    # 4. Ajout de l'index des poids comme colonne
    df.index.name = index_name
    df.reset_index(inplace=True)
    
    # 5. Application du style et du formatage
    styled_df = df.style.set_caption(title).format(
        # Applique un formatage flottant à toutes les colonnes numériques
        {col: '{:.6f}' for col in df.columns if col != index_name}
    ).set_properties(**{'text-align': 'center'})
    
    # 6. Affichage Console
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}\n")
    print(df.to_string(index=False)) 

    # 7. Sauvegarde en PNG (avec la correction max_cols=-1)
    try:
        dfi.export(styled_df, filename, max_cols=-1) 
        print(f"\n[OK] Tableau sauvegardé avec succès dans le fichier : **{filename}**")
        
    except Exception as e:
        print(f"\n[ERR] Erreur lors de l'exportation en PNG : {e}")

    return df

def display_and_save_weights_table(
    weights: Union[List[float], np.ndarray], 
    filename: str = "weights_table.png", 
    title: str = "Tableau des Poids Entraînés"
):
    """
    Crée un DataFrame pour un vecteur de poids, y ajoute l'index (Numéro du Poids),
    applique un formatage, l'affiche, et le sauvegarde en PNG.

    Args:
        weights (list ou np.ndarray): La liste ou le tableau NumPy des poids du modèle.
        filename (str): Le nom du fichier PNG à sauvegarder.
        title (str): Le titre du tableau.
    """
    
    # 1. Création du DataFrame
    # On crée le DataFrame à partir du vecteur des poids
    df = pd.DataFrame(weights, columns=['Valeur du Poids'])
    
    # 2. Ajout de l'index comme colonne 'Numéro du Poids'
    df.index.name = 'Numéro du Poids'
    df.reset_index(inplace=True)
    
    # 3. Application du style et du formatage (très important pour les poids flottants)
    styled_df = df.style.set_caption(title).format({
        # Formatage des poids avec une bonne précision
        'Valeur du Poids': '{:.6f}', 
        # Formatage de l'index sans décimale
        'Numéro du Poids': '{:,.0f}'
    }).set_properties(**{'text-align': 'center'})
    
    # 4. Affichage Console
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}\n")
    # Afficher le DataFrame sans l'index Pandas interne pour la console
    print(df.to_string(index=False)) 

    # 5. Sauvegarde en PNG
    try:
        import dataframe_image as dfi
        dfi.export(styled_df, filename) 
        print(f"\n[OK] Tableau sauvegardé avec succès dans le fichier : **{filename}**")
        
    except ImportError:
        print("\n[ERR] Erreur : La bibliothèque 'dataframe-image' n'est pas installée.")
        print("Veuillez l'installer avec : `pip install dataframe-image`")
    except Exception as e:
        print(f"\n[ERR] Erreur lors de l'exportation en PNG : {e}")
        print("Assurez-vous d'avoir un moteur de rendu (comme Chrome, Edge ou 'chromedriver') disponible.")

    return df

def display_and_save_results_table(
    results_list: List[Dict[str, Any]], 
    column_map: Dict[str, str], 
    filename: str = "results_table.png",
    title: str = "TABLEAU RÉCAPITULATIF DES RÉSULTATS"
):
    """
    Crée un DataFrame, applique un style, l'affiche, et le sauvegarde en PNG.
    L'adaptabilité est assurée par le paramètre 'column_map'.

    Args:
        results_list (List[Dict[str, Any]]): La liste des dictionnaires de résultats.
        column_map (Dict[str, str]): Un dictionnaire mappant les clés originales 
                                     (ex: 'Ea') aux noms d'affichage souhaités (ex: 'Erreur Apprentissage (Ea)').
        filename (str): Le nom du fichier PNG à sauvegarder.
        title (str): Le titre du tableau.
    """
    
    # 1. Créer le DataFrame
    df = pd.DataFrame(results_list)
    
    # 2. Renommer les colonnes basées sur le mapping fourni
    # On filtre le mapping pour ne garder que les clés réellement présentes dans le DataFrame
    effective_column_map = {k: v for k, v in column_map.items() if k in df.columns}
    df.rename(columns=effective_column_map, inplace=True)
    
    # 3. Déterminer l'ordre des colonnes basé sur le mapping
    # On utilise l'ordre des valeurs du mapping comme nouvel ordre
    column_order = [v for k, v in column_map.items() if k in effective_column_map]
    df = df[column_order]
    
    # 4. Appliquer le style
    styled_df = df.style.set_caption(title)
    
    # Définir les formats et l'alignement
    format_dict = {}
    center_subset = []
    
    # Définir des règles de formatage génériques
    for col in df.columns:
        if 'Erreur' in col or 'Err' in col or 'Ea' in col or 'Ev' in col or 'Et' in col:
            format_dict[col] = '{:.4f}'
            center_subset.append(col)
        elif 'Itérat' in col:
            format_dict[col] = '{:,.0f}'
            center_subset.append(col)
        elif 'Eta' in col or 'eta' in col:
            format_dict[col] = '{:.2f}'
            center_subset.append(col)
        else:
            center_subset.append(col)
            
    styled_df = styled_df.format(format_dict)
    styled_df = styled_df.set_properties(**{'text-align': 'center'}, subset=center_subset)
    styled_df = styled_df.background_gradient(cmap='YlGnBu', subset=[col for col in df.columns if 'Erreur' in col or 'Err' in col or 'Ea' in col or 'Ev' in col or 'Et' in col])

    # --- Affichage Console ---
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}\n")
    print(df.to_string(index=False)) 

    # --- Sauvegarde en PNG ---
    try:
        dfi.export(styled_df, filename) 
        print(f"\n[OK] Tableau sauvegardé avec succès dans le fichier : **{filename}**")
        
    except Exception as e:
        print(f"\n[ERR] Erreur lors de l'exportation en PNG. Assurez-vous que 'dataframe-image' est installé.")
        print(f"Détails de l'erreur : {e}")

    return df

def show(if_show=True):
    if if_show:
        plt.show()

def nl():
    print('-'*50, '\n')

def signe(x):
    if x > 0:
        return 1
    else:
        return -1

def compute_center(X):
    # Extract only coordinates (ignore bias at index 0)
    coords = np.array([p[1:] for p in X])
    center = np.mean(coords, axis=0)
    return center

def expand_universe_from_corner(X, factor=2.0):
    # 1. Extract only the coordinates (skipping the bias/label at index 0)
    coords_only = np.array([p[1:] for p in X])
    
    # 2. Find the "Bottom-Left" reference point
    # We find the minimum value for each dimension independently
    bottom_left = np.min(coords_only, axis=0)
    
    expanded_points = []

    for p in X:
        biais = p[0]
        coords = np.array(p[1:])

        # 3. Calculate distance from the bottom-left corner
        vector_from_corner = coords - bottom_left
        
        # 4. Expand ONLY along the first dimension (x-axis)
        # Multiplying the distance from the leftmost point by the factor
        vector_from_corner[0] = vector_from_corner[0] * factor
        
        # 5. Reconstruct the point relative to the corner
        new_pos = bottom_left + vector_from_corner
        expanded_points.append([biais] + new_pos.tolist())

    return expanded_points

def expand_universe_linear(X, factor=2.0):
    # We assume the 'coords' start from index 1 onwards
    # Extract only the coordinate parts to find the center
    coords_only = np.array([p[1:] for p in X])
    center = np.mean(coords_only, axis=0)
    
    expanded_points = []

    for p in X:
        biais = p[0]
        coords = np.array(p[1:])

        # Calculate the distance from center for all dimensions
        vector_from_center = coords - center
        
        # Only scale the FIRST dimension (index 0 of the coordinate part)
        # We keep the other dimensions (y, z, etc.) exactly as they were
        vector_from_center[0] = vector_from_center[0] * factor
        
        # Reconstruct the point
        new_pos = center + vector_from_center
        expanded_points.append([biais] + new_pos.tolist())

    return expanded_points

def create_points(nbPoints, N, bornes=(-10e6,10e6), seed=seed): #Ajout de 1.0 au début pour le biais

    rng = np.random.default_rng(seed)
    
    points = []
    borne_min, borne_max = bornes

    for p in range(nbPoints):
        point = []
        point.append(1.0) #Le biais

        for i in range(N):
            x_i = rng.uniform(borne_min, borne_max)
            point.append(x_i)
        
        points.append(point)

    return points

def init_perceptron_prof(N, biais_range=(-10e6, 10e6), seed=42):
  rng = np.random.default_rng(seed)

  w_prof = np.zeros(N+1)

  biais_min, biais_max = biais_range
  w_prof[0] = rng.uniform(biais_min, biais_max)
  for i in range(1,len(w_prof)):
      w_prof[i] = rng.uniform(biais_min, biais_max)
  return w_prof

def init_perceptron_prof_in_bounds(X, seed=42):
    rng = np.random.default_rng(seed)
    
    # 1. Convert X to array and strip the first column (assuming it's bias/label)
    # We want the actual spatial boundaries of the data
    coords = np.array([p[1:] for p in X])
    num_dim = coords.shape[1]  # N
    
    # 2. Calculate spatial boundaries
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    
    # 3. Create a random point P that is GUARANTEED to be inside the box
    point_on_boundary = rng.uniform(mins, maxs)
    
    # 4. Generate random spatial weights (the direction the teacher faces)
    # We use a smaller range (-1 to 1) because the bias will balance the scale
    w_spatial = rng.uniform(-1, 1, size=num_dim)
    
    # 5. Calculate the Bias (w_prof[0])
    # w0 + dot(w_spatial, point_on_boundary) = 0  =>  w0 = -dot(...)
    bias = -np.dot(w_spatial, point_on_boundary)
    
    # 6. Construct final vector: [bias, w1, w2, ..., wN]
    w_prof = np.zeros(num_dim + 1)
    w_prof[0] = bias
    w_prof[1:] = w_spatial
    
    return w_prof

def L_prof(X, w_prof):
  L_ens = []
  for x in X:
    L_ens.append([x,signe(np.dot(x, w_prof))])
  return L_ens

def make_L_non_LS(L, inversedExemple, seed):
    size_L = len(L)
    rng = np.random.default_rng(seed)
    
    random_indices = rng.choice(size_L, size=inversedExemple, replace=False)

    for idx in random_indices:
        exemple = L[idx]
        if exemple[1] == 1:
            exemple[1] = -1
        else:
            exemple[1] = 1

    return L

def X(n=2, d=2):
    """
    Génère toutes les combinaisons de `d` indices allant de 0 à n-1,
    sous forme de floats.
    """
    return [ [float(x) for x in t] for t in product(range(n), repeat=d) ]

def L (f, X_ens):
    L_ens = []
    for n in range(len(X_ens)):
      L_ens.append([X_ens[n],f(X_ens[n][1:])]) #Le biais (x_0 = 1) est exclus du calcul de la fonction booleenne
    return L_ens

def f_OU(x):
    n = len(x)
    f_x = -1

    for i in range(n):
      if x[i] == 1:
        f_x = 1
        return f_x
      else:
        pass
    return f_x


def f_AND(x):
    n = len(x)

    f_x = 1
    for i in range(n):
      if x[i] == 0:
        f_x = -1
        return f_x
      else:
        pass
    return f_x

def f_XOR(x):
      result = 0
      for bit in x:
          result ^= int(bit)  # XOR bit à bit
      return 1 if result == 1 else -1

def L (f, X_ens):
    L_ens = []
    for n in range(len(X_ens)):
      L_ens.append([X_ens[n],f(X_ens[n][1:])]) #Le biais (x_0 = 1) est exclus du calcul de la fonction booleenne
    return L_ens

def error (L, w):
    """Calcule l'erreur d'apprentissage du perceptron sur l'ensemble d'entraînement
    
    Args:
        - L : l'ensemble d'apprentissage
        - w : le perceptron entrainté
    """
    if isinstance(L, pd.DataFrame):
        print("[Avertissement] Conversion du DataFrame en Liste pour la fonction d'initialisation.")
        # Le DataFrame est converti en une liste de [features, target]
        L_converted = []
    
        for index, row in L.iterrows():
            L_converted.append([row['Donnee'], row['Classe voulue']])
            L = L_converted

    y = []
    X = []
    err = 0

    for exemple in L:
        X.append(exemple[0]) #Exemple (Features)
        y.append(exemple[1]) #Classe réelle

    # n()
    # print(range(len(X)))
    # n()

    for i in range(len(X)):
        # print('i : ', i)
        # print('y[i] : ',  y[i])
        # print('X[i] : ',  X[i])
        if (y[i] - signe(np.dot(w,X[i]))) != 0 :
            err += 1
         
    return err

def f_init_Hebb(L_ens, biais_range):
    if isinstance(L_ens, pd.DataFrame):
        L_ens_converted = []
        for index, row in L_ens.iterrows():
            L_ens_converted.append([row['Donnee'], row['Classe voulue']])
        L_ens = L_ens_converted
    
    p = len(L_ens)
    dim = len(L_ens[0][0])

    borne_inf = biais_range[0]
    borne_sup = biais_range[1]

    w = []
    for i in range(dim):
        w.append(0.0)

    # Initialiser le biais une seule fois
    w[0] = rd(borne_inf, borne_sup)  # Biais
    
    # Règle de Hebb pour les poids
    for k in range(p):
        w[0] = w[0] + (1.0 * L_ens[k][1])  # Mise à jour du biais
        for i in range(1, dim):
            w[i] = w[i] + L_ens[k][0][i] * L_ens[k][1]

    return w

def f_init_rand(L_ens, biais_range, seed=seed):
    if isinstance(L_ens, pd.DataFrame):
        print("[Avertissement] Conversion du DataFrame en Liste pour la fonction d'initialisation.")
        
        # Le DataFrame est converti en une liste de [features, target]
        L_ens_converted = []
        for index, row in L_ens.iterrows():
            L_ens_converted.append([row['Donnee'], row['Classe voulue']])
        L_ens = L_ens_converted
    
    rng = np.random.default_rng(seed)

    p = len(L_ens)
    dim = len(L_ens[0][0]) #This dimension assumes features have already been biaised 

    borne_min = biais_range[0]
    borne_max = biais_range[1]

    w = []
    for i in range(dim):
        w.append(0.0)

    biais = (rng.uniform(borne_min, borne_max))
    w[0] = biais #Ajout du biais
    for i in range(1,dim):
        w[i] = (rng.uniform(borne_min, borne_max))

    return w

def perceptron_online(w_vect, L_ens, eta, maxIter, err_train_stop=0.0):
    if isinstance(L_ens, pd.DataFrame):
        print("[Avertissement] Conversion du DataFrame en Liste pour la fonction d'initialisation.")

        # Le DataFrame est converti en une liste de [features, target]
        L_ens_converted = []
        for index, row in L_ens.iterrows():
            L_ens_converted.append([row['Donnee'], row['Classe voulue']])
        L_ens = L_ens_converted
        #print("in perceptron online")

    w_k = copy.deepcopy(w_vect)
    # print('perceptron avant entraintement : ', w_k)
    stop = 0
    err = []
    current_error = error(L_ens, w_k)  # Erreur initiale
    err.append(current_error)

    while stop < maxIter and current_error > err_train_stop:     # boucle sur les époques
        # print('\n'*2, '_'*20, '\n')
        # print("stop =", stop)
        
        # print('\n'*2, '_'*20, '\n')
        # print("current_error =", current_error)
        # print("err_train_stop =", err_train_stop)
        
        nb_mal_classe = 0

        for k in range(len(L_ens)):   # boucle sur les exemples
            # print('k : ', k)
            # print('perceptron à l étape k :', w_k)
            x_k = L_ens[k][0]
            t = L_ens[k][1]

            # produit scalaire
            w_k_scal_x_k = sum(w_k[i] * x_k[i] for i in range(len(x_k)))

            y = 1 if w_k_scal_x_k > 0 else -1
            #print("y = ", y)
            #print("t = ", t)
            if y != t:

                delta_w = eta * (t - y)
                for i in range(len(x_k)):
                    w_k[i] += delta_w * x_k[i]
                nb_mal_classe += 1

        #Ajout de l'erreur de l'époque
        current_error = error(L_ens, w_k)
        err.append(current_error)

        # fin de l'époque : vérification arrêt
        if nb_mal_classe == 0:
            #print("Convergence atteinte.")
            stop += 1
            break
        else:
            #print("nb_mal_classe : ", nb_mal_classe)
            stop += 1


    #print("in perceptron online - return")
    return w_k, stop-1, err

def perceptron_batch(w_vect, L_ens, eta, maxIter):
    if isinstance(L_ens, pd.DataFrame):
        print("[Avertissement] Conversion du DataFrame en Liste pour la fonction d'initialisation.")

        # Le DataFrame est converti en une liste de [features, target]
        L_ens_converted = []
        for index, row in L_ens.iterrows():
            L_ens_converted.append([row['Donnee'], row['Classe voulue']])
        L_ens = L_ens_converted
        #print("in perceptron online")  #print("in perceptron online")

    w_k = copy.deepcopy(w_vect)
    stop = 0

    delta_w = np.zeros(len(w_vect))

    err = []
    err.append(error(L_ens, w_vect))

    while stop < maxIter:     # boucle sur les époques
        #print('\n'*2, '_'*20, '\n')
        #print("stop =", stop)
        nb_mal_classe = 0
        err.append(error(L_ens, w_k))
        delta_w = np.zeros(len(w_vect))

        for k in range(len(L_ens)):   # boucle sur les exemples
            x_k = L_ens[k][0]
            t = L_ens[k][1]

            # produit scalaire
            w_k_scal_x_k = sum(w_k[i] * x_k[i] for i in range(len(x_k)))
            y = 1 if w_k_scal_x_k > 0 else -1

            #print("y = ", y)
            #print("t = ", t)

            if y != t:
                for i in range(len(x_k)):
                    delta_w[i] = delta_w[i] + eta * (t - y)*x_k[i]
                nb_mal_classe += 1

        # fin de l'époque : vérification arrêt
        w_k += delta_w

        if nb_mal_classe == 0:
            #print("Convergence atteinte.")
            stop += 1
            break
        else:
            #print("nb_mal_classe : ", nb_mal_classe)
            stop += 1

    #print("in perceptron batch - return")
    return w_k, stop-1, err

def draw_curve_and_save(
    data_list: Union[List[float], List[int]], 
    label_text: str, 
    plot_title: str, # Nouveau paramètre pour le titre
    filename: str,   # Nouveau paramètre pour la sauvegarde
    color: str = 'blue', 
    xlabel: str = 'Itérations/Époques', 
    ylabel: str = 'Erreur', 
    linewidth: int = 2,
    scatter_or_plot = 'scatter'
):
    """
    Crée une figure, trace une courbe d'erreur, configure les labels/titre,
    et sauvegarde le graphique dans un fichier.

    Args:
        data_list (list/np.ndarray): La liste des valeurs (Y).
        label_text (str): Le texte à utiliser pour la légende de la courbe.
        plot_title (str): Le titre principal du graphique.
        filename (str): Le nom du fichier de sortie (ex: 'courbe.png').
        color (str): La couleur de la ligne.
        xlabel (str): Label de l'axe X.
        ylabel (str): Label de l'axe Y.
        linewidth (int/float): L'épaisseur de la ligne.
    """
    
    # 1. Créer la figure et les axes (remplace le paramètre 'ax' qui n'est plus nécessaire)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # L'axe X est l'index de la liste (correspondant aux itérations/époques)
    iterations = range(1, len(data_list) + 1)
    
    # 2. Tracé de la ligne
    if scatter_or_plot == 'scatter':
        ax.scatter(iterations,
        data_list,
        label=label_text,
        color=color,
        linewidth=linewidth)
    else :
        ax.plot(iterations,
        data_list,
        label=label_text,
        color=color,
        linewidth=linewidth)
    
    # 3. Configuration des labels des axes, du titre et de la légende
    ax.set_title(plot_title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Ajout d'une grille pour la lisibilité
    ax.grid(True, linestyle='--', alpha=0.6)

    # Affichage de la légende
    ax.legend(loc='best', frameon=True)
    
    # 4. Sauvegarde du plot
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    print(f"[OK] Courbe sauvegardée avec succès dans le fichier : **{filename}**")
    
    # 5. Fermer la figure pour libérer la mémoire
    plt.close(fig)

def draw_curve(ax, list, label_text, color='black', xlabel='', ylabel='', linewidth=2):
    """
    Trace une courbe d'erreur (ou de performance) sur un objet Axes existant.

    Args:
        ax (matplotlib.axes.Axes): L'objet Axes 2D existant.
        error_list (list/np.ndarray): La liste des valeurs d'erreur/performance à tracer.
        label_text (str): Le texte à utiliser pour la légende de cette courbe.
        color (str): La couleur de la ligne (ex: 'blue', 'red', '#FF5733').
        linewidth (int/float): L'épaisseur de la ligne.
    """
    
    # L'axe X est l'index de la liste (0, 1, 2, ... correspondant aux itérations/époques)
    iterations = range(1, len(list) + 1)
    
    # Tracé de la ligne
    ax.plot(iterations, 
            list, 
            label=label_text, 
            color=color, 
            linewidth=linewidth)
    
    # Configuration des labels des axes (peut être fait ici si toujours les mêmes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Optionnel : Ajout d'une grille pour la lisibilité
    ax.grid(True, linestyle='--', alpha=0.6)

def draw_scatter_plot(ax, data_list, label_text, color='black', 
                      xlabel='', ylabel='', marker='o', s=50):
    # L'axe X est toujours l'index de la liste (correspondant aux itérations ou numéros d'échantillon)
    x_indices = range(1, len(data_list) + 1)
    
    # Tracé des points (scatter plot)
    ax.scatter(x_indices, 
               data_list, 
               label=label_text, 
               color=color, 
               marker=marker, 
               s=s,
               alpha=0.7) # Ajout d'une légère transparence pour les points

    # Configuration des labels des axes (en utilisant les paramètres passés)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Optionnel : Ajout d'une grille pour la lisibilité
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_L_et_w_bias_first(L_ens, w_k):
    """
    L_ens: liste de tuples (x, t) où x = [1, x1, x2] (biais en premier)
    w_k: [w0, w1, w2] où w0 est le biais
    """

    print("CKPOINT 1")
    n()
    print("L_ens:", L_ens)
    n()
    print("CKPOINT 1")
    # Extraire les points (supporte x = [1,x1,x2] ou x = [x1,x2])
    x1_vals = []
    x2_vals = []
    labels = []
    for x, t in L_ens:
        labels.append(t)
        lx = len(x)
        if lx >= 3:
            x1_vals.append(x[1])
            x2_vals.append(x[2])
        elif lx == 2:
            x1_vals.append(x[0])
            x2_vals.append(x[1])
        else:
            raise ValueError(f"Unexpected feature vector length: {lx} for x={x}")

    # Créer le plot
    plt.figure(figsize=(8, 6))

    # Tracer les points
    colors = ['red' if t == -1 else 'blue' for t in labels]
    plt.scatter(x1_vals, x2_vals, c=colors, s=100, alpha=0.7,
                label=f'Classe -1 (rouge) / +1 (bleu)')

    # Tracer la droite de décision
    if len(w_k) < 3:
        raise ValueError(f"Perceptron weight vector must have length >=3, got {len(w_k)}: {w_k}")
    w0, w1, w2 = w_k[0], w_k[1], w_k[2]

    # Générer des points pour la droite
    x1_min, x1_max = min(x1_vals), max(x1_vals)

    if abs(w2) > 1e-10:  # w2 ≠ 0
        # w1*x1 + w2*x2 + w0 = 0 → x2 = -(w1*x1 + w0)/w2
        x1_line = np.array([x1_min - 1, x1_max + 1])
        x2_line = -(w1 * x1_line + w0) / w2
    else:  # droite verticale
        x_fixed = -w0 / w1
        x2_min, x2_max = min(x2_vals), max(x2_vals)
        x1_line = np.array([x_fixed, x_fixed])
        x2_line = np.array([x2_min - 1, x2_max + 1])

    plt.plot(x1_line, x2_line, 'k-', linewidth=3, label=f'{w1:.2f}·x1 + {w2:.2f}·x2 + {w0:.2f} = 0')

    # Ajouter des indications
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Perceptron - Frontière de décision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Afficher les poids
    print(f"Équation de la droite: {w1:.3f}·x1 + {w2:.3f}·x2 + {w0:.3f} = 0")
    print(f"Vecteur normal (orthogonal à la droite): ({w1:.3f}, {w2:.3f})")
    print(f"Vecteur directeur (le long de la droite): ({-w2:.3f}, {w1:.3f})")

    plt.show()
    return w0, w1, w2

def plot_L_et_w_multiple_fixed_axes(L_ens, W_array, xlim=None, ylim=None):
    """
    Version avec limites x et y séparées.

    xlim: tuple (x_min, x_max) ou None
    ylim: tuple (y_min, y_max) ou None
    """

    # Extraire points (supporte x = [1,x1,x2] ou x = [x1,x2])
    x1_vals = []
    x2_vals = []
    labels = []
    for x, t in L_ens:
        labels.append(t)
        lx = len(x)
        if lx >= 3:
            x1_vals.append(x[1])
            x2_vals.append(x[2])
        elif lx == 2:
            x1_vals.append(x[0])
            x2_vals.append(x[1])
        else:
            raise ValueError(f"Unexpected feature vector length: {lx} for x={x}")

    # Graphique
    plt.figure(figsize=(10, 8))

    # Points
    colors = ['red' if t == -1 else 'blue' for t in labels]
    plt.scatter(x1_vals, x2_vals, c=colors, s=100, alpha=0.7,
                label='Classe -1 (rouge) / +1 (bleu)')

    # Limites automatiques si non spécifiées
    if xlim is None:
        x_min, x_max = min(x1_vals), max(x1_vals)
        x_margin = (x_max - x_min) * 0.2
        xlim = (x_min - x_margin, x_max + x_margin)

    if ylim is None:
        y_min, y_max = min(x2_vals), max(x2_vals)
        y_margin = (y_max - y_min) * 0.2
        ylim = (y_min - y_margin, y_max + y_margin)

    # Étendue pour calculer les droites (plus grande que l'affichage)
    x_line_range = np.array([xlim[0] - 10, xlim[1] + 10])

    # Tracer chaque droite
    num_vectors = len(W_array)
    colors_vectors = plt.cm.rainbow(np.linspace(0, 1, max(2, num_vectors)))

    for i, w in enumerate(W_array):
        wa = np.asarray(w).ravel()
        # Accept weight vectors with bias ([w0,w1,w2]) or without ([w1,w2])
        if wa.size >= 3:
            w0, w1, w2 = float(wa[0]), float(wa[1]), float(wa[2])
        elif wa.size == 2:
            w0 = 0.0
            w1, w2 = float(wa[0]), float(wa[1])
        else:
            raise ValueError(f"Weight vector must have 2 or 3 elements, got {wa}")

        if abs(w2) > 1e-10:
            y_line = -(w1 * x_line_range + w0) / w2
        else:
            x_fixed = -w0 / w1
            x_line_range = np.array([x_fixed, x_fixed])
            y_line = np.array([ylim[0] - 10, ylim[1] + 10])

        if i == 0:
            plt.plot(x_line_range, y_line, 'r-', linewidth=3, label='Professeur')
        else:
            plt.plot(x_line_range, y_line, color=colors_vectors[i],
                     linewidth=1.5, linestyle='--', label=f'Élève {i}')

    # FORCER les limites
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    # Origine
    plt.scatter([0], [0], color='green', s=100, marker='*', label='Origine')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Perceptron - {num_vectors} vecteurs (axes fixés)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def pocket_algorithm(L_ens, w_init, eta, max_iter=1000, target_error=None):
    """Algorithme Pocket: garde le meilleur résultat du perceptron
    
    Args:
        L_ens: ensemble d'apprentissage (DataFrame ou liste)
        w_init: poids initiaux
        eta: pas d'apprentissage (delta)
        max_iter: nombre maximum d'itérations
        target_error: erreur cible pour arrêter l'apprentissage (None = pas d'arrêt)
    
    Returns:
        w_best: meilleur vecteur de poids trouvé
        best_error: meilleure erreur trouvée
        iterations: nombre d'itérations effectuées
        error_history: historique des erreurs
    """
    if isinstance(L_ens, pd.DataFrame):
        L_converted = []
        for index, row in L_ens.iterrows():
            L_converted.append([row['Donnee'], row['Classe voulue']])
        L_ens = L_converted
    
    w = copy.deepcopy(w_init)
    w_best = copy.deepcopy(w_init)
    best_error = error(L_ens, w_best)
    error_history = [best_error]
    
    iterations = 0
    
    for epoch in range(max_iter):
        nb_mal_classe = 0
        for k in range(len(L_ens)):
            x_k = L_ens[k][0]
            t = L_ens[k][1]
            
            w_k_scal_x_k = sum(w[i] * x_k[i] for i in range(len(x_k)))
            y = 1 if w_k_scal_x_k > 0 else -1
            
            if y != t:
                delta_w = eta * (t - y)
                for i in range(len(x_k)):
                    w[i] += delta_w * x_k[i]
                nb_mal_classe += 1
        
        # Calculer l'erreur actuelle
        current_error = error(L_ens, w)
        error_history.append(current_error)
        
        # Mise à jour de la "poche" si meilleure
        if current_error < best_error:
            w_best = copy.deepcopy(w)
            best_error = current_error
        
        iterations += 1
        
        # Arrêt si erreur cible atteinte
        if target_error is not None and best_error <= target_error:
            break
        
        # Arrêt si convergence
        if nb_mal_classe == 0:
            break
    
    return w_best, best_error, iterations, error_history
    
def stabilite(w, x, tau):
    return (tau*np.dot(w,x))/np.linalg.norm(w)

def stabilites(L, w):
    """ Calcul de la distance des exemples à l'hyperplan séparateur """
    
    if isinstance(L, pd.DataFrame):
        print("[Avertissement] Conversion du DataFrame en Liste pour la fonction d'initialisation.")
        # Le DataFrame est converti en une liste de [features, target]
        L_converted = []
    
        for index, row in L.iterrows():
            L_converted.append([row['Donnee'], row['Classe voulue']])
        L = L_converted

    y = []
    X = []
    stabilites = []

    for exemple in L:
        X.append(exemple[0]) #Exemple (Features)
        y.append(exemple[1]) #Classe réelle
    
    for i in range(len(X)):
        stabilites.append([X[i], stabilite(w, X[i], y[i])])# print('i : ', i)

    
    return stabilites

def parse_custom_file(path: str, n: int):
    # Lire tout le fichier
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1) Sauter les n premières lignes (n incluse)
    lines = lines[n:]  

    # 2) Rejoindre les lignes en un seul texte
    content = "".join(lines)

    # 3) Découper sur la séquence ' \n'
    # On nettoie aussi les éléments vides
    raw_entries = [block.strip() for block in content.split(" \n") if block.strip()]

    # 4) Construire le dictionnaire final
    result = {}
    for entry in raw_entries:
        # Vérifier qu'il y a bien un séparateur ':'
        if ":" not in entry:
            print(f"[AVERTISSEMENT] Pas de ':' dans : {entry}")
            continue
        
        key, value = entry.split(":", 1)  # 1 → on sépare seulement au premier ':'
        key = key.strip()
        value = value.strip()
        result[key] = value

    return result

def clean_sonar_dict(raw_dict):
    cleaned = {}
    for key, value in raw_dict.items():
        # ... (votre nettoyage initial)
        value = value.strip("{} \n")
        value = value.replace("{", "").replace("}", "").strip()

        # Le problème est probablement ici: value.split() ne gère pas
        # un cas où deux échantillons sont fusionnés à cause d'un ' \n' manquant.
        
        # Solution: On nettoie TOUS les caractères non numériques/non décimaux/non espace
        # pour s'assurer que seuls les nombres et les séparateurs d'espace subsistent.
        import re
        # Remplacer tout ce qui n'est PAS un chiffre, un point, ou un espace par un simple espace
        # Cela devrait éliminer les accolades résiduelles, les retours à la ligne indésirables, etc.
        value = re.sub(r'[^\d\.\s]', ' ', value)
        
        # Séparer par tout type d'espace et filtrer les chaînes vides
        nums = [float(x) for x in value.split() if x] # Convertir directement en float
        
        if len(nums) != 60:
             print(f"[ALERTE] Échantillon {key} a {len(nums)} valeurs (devrait être 60).")
        
        cleaned[key] = nums
    return cleaned

def parse_sonar_file(path):
    data = []
    line_lengths = []

    with open(path, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            parts = line.split()
            line_lengths.append((line_number, len(parts)))
            data.append(parts)

    # Print dimension info
    print("\n=== Dimension analysis ===")
    lengths = [length for _, length in line_lengths]
    unique_lengths = set(lengths)

    print(f"Unique line lengths found: {unique_lengths}")

    if len(unique_lengths) > 1:
        print("\nWARNING: Lines have inconsistent numbers of values!\n")
        for line_number, length in line_lengths:
            if length != max(unique_lengths):
                print(f" → Line {line_number} has {length} values (expected {max(unique_lengths)})")

    return {
        "data": data,
        "line_lengths": line_lengths
    }

def early_stopping(L_ens, L_val, w_init, eta, max_iter=1000, patience=10):
    """Early Stopping: arrête l'apprentissage quand l'erreur de validation augmente
    
    Args:
        L_ens: ensemble d'apprentissage
        L_val: ensemble de validation
        w_init: poids initiaux
        eta: pas d'apprentissage
        max_iter: nombre maximum d'itérations
        patience: nombre d'itérations sans amélioration avant arrêt
    
    Returns:
        w_best: meilleur vecteur de poids (sur validation)
        best_val_error: meilleure erreur de validation
        iterations: nombre d'itérations effectuées
        train_errors: historique des erreurs d'apprentissage
        val_errors: historique des erreurs de validation
    """
    if isinstance(L_ens, pd.DataFrame):
        L_ens_converted = []
        for index, row in L_ens.iterrows():
            L_ens_converted.append([row['Donnee'], row['Classe voulue']])
        L_ens = L_ens_converted
    
    if isinstance(L_val, pd.DataFrame):
        L_val_converted = []
        for index, row in L_val.iterrows():
            L_val_converted.append([row['Donnee'], row['Classe voulue']])
        L_val = L_val_converted
    
    w = copy.deepcopy(w_init)
    w_best = copy.deepcopy(w_init)
    best_val_error = error(L_val, w_best)
    
    train_errors = [error(L_ens, w_init)]
    val_errors = [best_val_error]
    
    iterations = 0
    no_improvement = 0
    
    for epoch in range(max_iter):
        # Une époque d'apprentissage
        for k in range(len(L_ens)):
            x_k = L_ens[k][0]
            t = L_ens[k][1]
            
            w_k_scal_x_k = sum(w[i] * x_k[i] for i in range(len(x_k)))
            y = 1 if w_k_scal_x_k > 0 else -1
            
            if y != t:
                delta_w = eta * (t - y)
                for i in range(len(x_k)):
                    w[i] += delta_w * x_k[i]
        
        # Calculer les erreurs
        train_err = error(L_ens, w)
        val_err = error(L_val, w)
        
        train_errors.append(train_err)
        val_errors.append(val_err)
        
        # Mise à jour du meilleur modèle
        if val_err < best_val_error:
            w_best = copy.deepcopy(w)
            best_val_error = val_err
            no_improvement = 0
        else:
            no_improvement += 1
        
        iterations += 1
        
        # Early stopping
        if no_improvement >= patience:
            break
        
        # Arrêt si convergence sur train
        if train_err == 0:
            break
    
    return w_best, best_val_error, iterations, train_errors, val_errors

def plot_stabilities(stabilities, title="Stabilités des exemples"):
    """Trace un graphique des stabilités"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(stabilities)), stabilities, 'b-', marker='o', markersize=3)
    plt.axhline(y=0, color='r', linestyle='--', label='Hyperplan séparateur')
    plt.xlabel('Index de l\'exemple')
    plt.ylabel('Stabilité (gamma)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def print_weights(w, title="Poids du perceptron"):
    """Affiche les poids du perceptron"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Biais (w[0]): {w[0]:.6f}")
    print(f"\nPoids des features (w[1] à w[{len(w)-1}]):")
    for i in range(1, len(w)):
        print(f"  w[{i}] = {w[i]:.6f}")
    print(f"{'='*60}\n")

def import_data():
    raw_dict = parse_custom_file('./sonar.mines', 9)
    mines_dict = clean_sonar_dict(raw_dict)

    raw_dict = parse_custom_file('./sonar.rocks', 8) #Ce fichier n'a que 8 lignes au début
    rocks_dict = clean_sonar_dict(raw_dict)

    data=[]

    for key, values in mines_dict.items():
        sample_entry = {
                'ID': key,              # Le nom de l'échantillon
                'Donnee': values,       # La liste des 60 valeurs (PAS besoin de re-chercher dans le dict)
                'Classe voulue': 'M'    # L'étiquette de classe
            }
        data.append(sample_entry)

    for key, values in rocks_dict.items():
        sample_entry = {
                'ID': key,              # Le nom de l'échantillon
                'Donnee': values,       # La liste des 60 valeurs (PAS besoin de re-chercher dans le dict)
                'Classe voulue': 'R'    # L'étiquette de classe
            }
        data.append(sample_entry)

    data_df = pd.DataFrame(data)

    #Ajout du biais aux entrées (toujours = 1)
    for index, row in data_df.iterrows():
        data = row['Donnee'] 
        data_with_biais = [1.0] + data
        data_df.at[index, 'Donnee'] = data_with_biais

    #Séparation des données en ensemble de test et de train
    train_list = []
    test_list = []

    for index, row in data_df.iterrows():
        if '*' in row['ID']:
            train_list.append(row)
        else:
            test_list.append(row)

    train_df = pd.DataFrame(train_list)
    test_df = pd.DataFrame(test_list)

    return data_df, test_df, train_df

def pretraitement(test_df, train_df):
    # 1 - Remove Col ID
    test_df_prepare = test_df.drop('ID', axis=1)
    train_df_prepare = train_df.drop('ID', axis=1)

    # 2 - Map classe voulue to -1 / +1
    mapping_classes = {'M': 1, 'R': -1}
    train_df_prepare['Classe voulue'] = train_df_prepare['Classe voulue'].map(mapping_classes)
    test_df_prepare['Classe voulue'] = test_df_prepare['Classe voulue'].map(mapping_classes)
    
    return train_df_prepare,test_df_prepare


def expand_universe(L_ens, w, delta):
    """
    Pousse les points loin de l'hyperplan w par un facteur delta.
    Chaque point x_mu devient x_mu + delta * y_mu * (w / ||w||)
    """
    # 1. Normaliser w pour avoir la direction pure (le vecteur unité)
    w_norm = np.asarray(w) / np.linalg.norm(w)
    
    L_expanded = []
    
    # 2. Parcourir les exemples (on suit votre structure [Donnee, Classe voulue])
    for exemple in L_ens:
        x_mu = np.asarray(exemple[0], dtype=float)
        y_mu = float(exemple[1])
        
        # Pousser le point : le signe de y_mu assure qu'il part du bon côté
        # Si y=1, on va dans le sens de w. Si y=-1, on va à l'opposé.
        x_pushed = x_mu + (delta * y_mu * w_norm)
        
        # Conserver le format d'origine pour être compatible avec vos fonctions
        L_expanded.append([x_pushed.tolist(), y_mu])
        
    return L_expanded

def minim_error(L, w, maxIter, eta, temp, temp_variable=False):
    """
    Implements the Minimerror learning rule.
    Minimizes the cost function Somme sur mu de V(mu) = [1 - tanh(beta * gamma / 2)] / 2
    """
    
    # Convert to numpy arrays for safe numeric ops
    X = [np.asarray(ex[0], dtype=float) for ex in L]
    y = [float(ex[1]) for ex in L]

    w = np.asarray(w, dtype=float)
    beta = 1.0 / float(temp)
    w_k = []
    
    w_k.append(copy.deepcopy(w))
    for _ in range(int(maxIter)):
        print("w avant mise à jour:", w)
        for mu in range(len(X)):
            gamma = stabilite(w, X[mu], y[mu])
            arg = (beta * gamma) / 2.0

            # derivative uses sech^2 = 1 - tanh^2(arg)
            # sech2 = 1.0 - np.tanh(arg)**2

            # gradient is scalar * X[mu]
            grad_scalar = - (beta / 4.0) * 1/(np.cosh(arg)**2) * y[mu]
            gradient_factor = grad_scalar * X[mu]

            w = w - eta * gradient_factor
        w = w / np.linalg.norm(w)
        print("w après mise à jour:", w)
        w_k.append(copy.deepcopy(w))
    # print("Weights evolution during Minimerror training:")
    # print(w_k)
    return w_k

def minim_error_avec_recuit(L_ens, w, maxIter, eta, 
                            temp, temp_target,
                            grad_stop_pos, grad_stop_neg):
    
    if isinstance(L_ens, pd.DataFrame):
        L_ens_converted = []
        for index, row in L_ens.iterrows():
            L_ens_converted.append([row['Donnee'], row['Classe voulue']])
        L_ens = L_ens_converted

    X = [np.asarray(ex[0], dtype=float) for ex in L_ens]
    y = [float(ex[1]) for ex in L_ens]
    w = np.asarray(w, dtype=float)

    print("In Minimerror")
    # print("L_ens : ", L_ens)
    # print("X : ", X)
    # print("y : ", y)
    # print("w : ", w)
    


    w_history = []
    temp_history = []
    
    # Store initial state
    w_history.append(w.copy())
    temp_history.append(temp)
    
    gamme_one_iter = []
    gamma_history = []
    gamma_history.append(0)
    
    err = error(L_ens, w)
    error_history = []
    error_history.append(err)

    temp_decay_factor = (temp_target / temp) ** (1.0/maxIter)
    t_k = temp
    
    iter = 1
    while iter <= maxIter and err != 0:
        err = error(L_ens, w)
        beta_k = 1.0 / float(t_k)
        if iter % 25 == 0:
            print(f"Iteration {iter+1}/{int(maxIter)}, Température t_k: {t_k}, Beta_k: {beta_k}")
        for mu in range(len(X)):
            gamma = stabilite(w, X[mu], y[mu])
            gamme_one_iter.append(gamma)
            arg = (beta_k * gamma) / 2.0
            if arg > grad_stop_pos:
                # print("mu", mu)
                # print("arg > ", grad_stop_pos, " - Skipped")
                grad_scalar = 0.0
            elif arg < grad_stop_neg:
                # print("mu", mu)
                # print("arg < ", grad_stop_neg, " - Skipped")
                grad_scalar = 0.0
            else:
                grad_scalar = - (beta_k / 4.0) * (1.0 / (np.cosh(arg)**2)) * y[mu]
                # print("mu", mu)
                # print("grad_scalar : ", grad_scalar)


            w = w - eta * (grad_scalar * X[mu])

        t_k = t_k * temp_decay_factor  
        # w = w / np.linalg.norm(w)
        iter += 1    


        # Record state AFTER the update
        w_history.append(w.copy())
        temp_history.append(t_k)
        gamma_history.append(np.mean(gamme_one_iter))
        
        err = error(L_ens, w)
        error_history.append(err)
    
    return w_history, temp_history, gamma_history, error_history

#Minim error with two different temperatures
def minim_error_2_temp(L_ens, w_init, maxIter, eta, temp_init, ratio_beta=1.0):
    """
    Minimerror avec 2 températures (Beta+ et Beta-).
    
    Args:
        L_ens: Liste des exemples d'apprentissage
        w_init: Poids initiaux
        maxIter: Nombre max d'itérations
        eta: Pas d'apprentissage
        temp_init: Température de base (T) pour les erreurs (Beta-)
        ratio_beta: Rapport Beta+ / Beta-. 
                    Si ratio > 1, on est plus "froid" (strict) sur les exemples corrects.
                    Si ratio < 1, on est plus "chaud" (souple) sur les exemples corrects.
    """
    gamma_one_iter = []
    gamma_history = []

    # 1. Préparation des données
    if isinstance(L_ens, pd.DataFrame):
        L_temp = []
        for index, row in L_ens.iterrows():
            L_temp.append([row['Donnee'], row['Classe voulue']])
        L_ens = L_temp

    X = [np.asarray(ex[0], dtype=float) for ex in L_ens]
    y = [float(ex[1]) for ex in L_ens]
    w = np.asarray(w_init, dtype=float)
    
    # Pour le calcul de l'erreur
    L_data_simple = list(zip(X, y))
    
    # 2. Historiques
    w_history = [w.copy()]
    # On stocke T_base (T_) pour l'historique
    temp_history = [temp_init] 
    err_init = error(L_data_simple, w)
    error_history = [err_init]
    
    # 3. Paramètres de Recuit (Optionnel, ici on fait un recuit simple sur T_base)
    t_k = temp_init
    # On vise une température finale assez basse pour converger
    t_target = 0.1 
    decay = (t_target / t_k) ** (1.0 / maxIter)

    print(f"Démarrage Minimerror 2T. T_init={t_k}, T_target={t_target}, Ratio Beta+/Beta-={ratio_beta}")

    for k in range(int(maxIter)):
        
        # Calcul des deux Bêtas
        # Beta_minus (pour les erreurs, gamma < 0)
        beta_minus = 1.0 / t_k
        
        # Beta_plus (pour les exemples appris, gamma > 0)
        # D'après la formule : Beta+ / Beta- = ratio
        beta_plus = beta_minus * ratio_beta
        
        # Pour le diagnostic
        if k % 100 == 0:
            print(f"Iter {k}: T={t_k:.3f}, B-={beta_minus:.2f}, B+={beta_plus:.2f}, Err={error_history[-1]}")

        # Mise à jour stochastique
        for mu in range(len(X)):
            # Calcul de la stabilité de l'exemple mu
            gamma = stabilite(w, X[mu], y[mu])
            gamma_one_iter.append(gamma)
            # --- SELECTION DE LA TEMPERATURE ---
            if gamma >= 0:
                current_beta = beta_plus  # Exemple bien classé
            else:
                current_beta = beta_minus # Exemple mal classé (Erreur)
            # -----------------------------------

            arg = (current_beta * gamma) / 2.0
            
            # Limites numériques pour éviter l'overflow/underflow
            if arg > 15: # Trop stable, gradient ~ 0
                grad_scalar = 0.0
            elif arg < -15: # Trop instable, gradient ~ 0 (optionnel)
                grad_scalar = 0.0
            else:
                # Gradient de V(gamma)
                # Minimerror minimise V, donc w(t+1) = w(t) - eta * grad
                # grad_V = - (beta/4) * sech^2(arg)
                # Le signe 'moins' dans la formule du gradient scalaire compense la descente
                # Note: Dans la formule physique, on ajoute le terme qui augmente la stabilité.
                
                deriv = (current_beta / 4.0) * (1.0 / (np.cosh(arg)**2))
                # On multiplie par y[mu] car d(gamma)/dw est proportionnel à y*x
                grad_scalar = -deriv * y[mu]

            # Mise à jour : w = w - eta * gradient
            # gradient_w = grad_scalar * x
            # w = w - eta * (grad_scalar * x) 
            # => w = w + eta * deriv * y * x
            w = w - eta * (grad_scalar * X[mu])

        # Normalisation (Condition Minimerror ||J||=cste)
        w_norm = np.linalg.norm(w)
        if w_norm > 0:
            w = w / w_norm

        # Mise à jour du recuit (on refroidit T_base)
        t_k = t_k * decay
        
        # Enregistrement
        w_history.append(w.copy())
        temp_history.append(t_k)
        error_history.append(error(L_data_simple, w))
        gamma_history.append(np.mean(gamma_one_iter))
        # Arrêt précoce si 0 fautes (optionnel, souvent on continue pour maximiser la marge)
        if error_history[-1] == 0 and k > maxIter/2:
            break

    return w_history, temp_history, gamma_history, error_history




f = False
t = True

# if_import_data = f
if_import_data = t

if_question_1_1 = f
# if_question_1_1 = t

if_question_1_2 = f
# if_question_1_2 = t

if_question_1_3 = f
# if_question_1_3 = t


if_question_2_0 = f
# if_question_2_0 = t

if_question_2_1 = f
# if_question_2_1 = t

if_question_2_2 = f
# if_question_2_2 = t

# if_question_2_3 = f
if_question_2_3 = t

if __name__=="__main__":

    if if_import_data:
        #IMPORTATION DONNEES
        data_df, test_df, train_df = import_data()

        #PRETRAITEMENT
        train_df_prepare, test_df_prepare = pretraitement(test_df, train_df)
    
    if if_question_1_1:
        print("-"*25, "QUESTION 1_1", "-"*25)
        print("Minimerror avec recuit deterministe - Entraintement sur L_Train et Test sur L_Test")
        print("train_df_prepare")
        print(train_df_prepare)
        print("test_df_prepare")
        print(test_df_prepare)
        nl()

        w = np.array(f_init_Hebb(train_df_prepare, [1000, 1000]))
        w = w / np.linalg.norm(w)

        print("Initial weights w:", w)


        nl()
        maxIter = 4000

        eta = 0.0001

        temp=0.25
        temp_target = 0.001

        grad_stop_pos = 0
        grad_stop_neg = -10

        print(f"eta: {eta}")
        print(f"maxIter: {maxIter}")
        
        print(f"temp départ: {temp}")
        print(f"temp_target: {temp_target}")
        print("Règle de recuit : Décroissance exponentielle : temp_decay_factor = (temp_target / temp) ** (1.0/maxIter)")
        
        print(f"Seuil de hard cut numérique pour le calcul du gradient (> 0) : {grad_stop_pos}")
        print(f"Seuil de hard cut numérique pour le calcul du gradient (< 0) : {grad_stop_neg}")
        nl()
        

        # Unpack the two lists directly
        weights, temps, gammas, errors = minim_error_avec_recuit(train_df_prepare, w, 
                                                         maxIter=maxIter, eta=eta,
                                                         temp=temp, temp_target=temp_target,
                                                         grad_stop_pos=grad_stop_pos,
                                                         grad_stop_neg=grad_stop_neg)

        weight_norms = [np.linalg.norm(w) for w in weights]

        viz = PerceptronVisualizer(train_df_prepare, weights, 
                            (temps, "Temperature"), 
                            (gammas, "Mean Gamma"),
                            (errors, "Error"),
                            (weight_norms, "Weight Norm"),
                            show_plot=False)
        

        
        viz.save_tracks_separately(prefix="Q1_1/Q1_1 - ")

        stabilites_train = []
        stabilites_paires = stabilites(train_df_prepare, weights[-1])
        
        for paire in stabilites_paires:
            stabilites_train.append(paire[1])
        
        draw_curve_and_save(
            stabilites_train, 
            label_text='Stabilité', 
            plot_title="Stabilité gamma sur Ensemble d entrainement (L_train)", 
            filename='Q1_1/Q1_1 - Stabilité sur ensemble d entrainement',
            color='blue', 
            xlabel='Patrons', 
            ylabel='Stabilité gamma sur Ensemble d entrainement', 
            linewidth=2,
            scatter_or_plot='scatter'
        )

        df_weights = display_and_save_weights_table(
            weights[-1],
            filename="Q1_1/Q1_1_Poids_Perceptron.png",
            title="Q1_1 - Poids du Perceptron après Entraînement"
        )

        err = error(train_df_prepare, weights[-1])
        print("last erreur d'apprentissage minimerror : ", err)
        min_err = min(errors)
        best_iteration = errors.index(min_err)   

        print(f"Minimum error found by Minimerror: {min_err} at iteration {best_iteration}")

        nl()
        print("-"*25, "TEST SUR test_df_prepare", "-"*25)
        E_g = error(test_df_prepare, weights[-1])

        print("Erreur de généralisation :", E_g)
        
    if if_question_1_2:
        print("-"*25, "QUESTION 1_2", "-"*25)
        print("Minimerror avec recuit deterministe - Entraintement sur L_Train+L_Test")
        complete_df_prepare = pd.concat([train_df_prepare, test_df_prepare], ignore_index=True)
    
        w = np.array(f_init_Hebb(complete_df_prepare, [1000, 1000]))
        w = w / np.linalg.norm(w)

        print("Initial weights w:", w)


        nl()
        maxIter = 5000

        eta = 0.0001

        temp=0.25
        temp_target = 0.001

        grad_stop_pos = 0
        grad_stop_neg = -10


        print(f"eta: {eta}")
        print(f"maxIter: {maxIter}")
        
        print(f"temp départ: {temp}")
        print(f"temp_target: {temp_target}")
        print("Règle de recuit : Décroissance exponentielle : temp_decay_factor = (temp_target / temp) ** (1.0/maxIter)")
        
        print(f"Seuil de hard cut numérique pour le calcul du gradient (> 0) : {grad_stop_pos}")
        print(f"Seuil de hard cut numérique pour le calcul du gradient (< 0) : {grad_stop_neg}")
        nl()

        # Unpack the two lists directly
        weights, temps, gammas, errors = minim_error_avec_recuit(complete_df_prepare, w, 
                                                         maxIter=maxIter, eta=eta,
                                                         temp=temp, temp_target=temp_target,
                                                         grad_stop_pos=grad_stop_pos,
                                                         grad_stop_neg=grad_stop_neg)

        weight_norms = [np.linalg.norm(w) for w in weights]

        viz = PerceptronVisualizer(complete_df_prepare, weights, 
                            (temps, "Temperature"), 
                            (gammas, "Mean Gamma"),
                            (errors, "Error"),
                            (weight_norms, "Weight Norm"),
                            show_plot=False)
        

        
        viz.save_tracks_separately(prefix="Q1_2/Q1_2 - Train+Test - ")

        stabilites_all = []
        stabilites_paires = stabilites(complete_df_prepare, weights[-1])
        
        for paire in stabilites_paires:
            stabilites_all.append(paire[1])
        
        draw_curve_and_save(
            stabilites_all, 
            label_text='Stabilité', 
            plot_title="Stabilité gamma sur Ensembles d entrainement + test (L_train + L_test)", 
            filename='Q1_2/Q1_2 - Stabilité sur ensembles d entrainement + test',
            color='blue', 
            xlabel='Patrons', 
            ylabel='Stabilité gamma sur Ensembles d entrainement + test', 
            linewidth=2,
            scatter_or_plot='scatter'
        )

        df_weights = display_and_save_weights_table(
            weights[-1],
            filename="Q1_2/Q1_2_Poids_Perceptron.png",
            title="Q1_2 - Poids du Perceptron après Entraînement"
        )

        err = error(complete_df_prepare, weights[-1])
        print("last erreur d'apprentissage minimerror sur Ensembles d'entrainement + test : ", err)
        min_err = min(errors)
        best_iteration = errors.index(min_err)   

        print(f"Minimum error found by Minimerror: {min_err} at iteration {best_iteration}")

    if if_question_2_0:
        print("-"*25, "QUESTION 2_0", "-"*25)
        print("Minimerror avec deux températures Beta+ / Beta- - Test sur un ensemble aléatoire garanti LS avec technique du Perceptron Professeur")
        seed_1 = 45
        seed_2 = 99
        seed_3 = 21

        nbPoints = 200
        N = 2
        bornes = [-20,20]
        print("Points générés pour nbPoints =", nbPoints, ", N =", N, ", bornes =", bornes, ", seed =", seed)
        X_ens = create_points(nbPoints, N, bornes, seed_1)

        w_prof =init_perceptron_prof_in_bounds(X_ens, seed_2)


        L_ens = L_prof(X_ens, w_prof)

       
        w = np.array(f_init_Hebb(L_ens, [1000, 1000]))
        w = w / np.linalg.norm(w)
        print("Initial weights w:", w)

        nl()
        
        maxIter = 1000
        eta = 0.001
        temp=0.25
        ratio_temperature = 1.0

        print(f"eta: {eta}")
        print(f"maxIter: {maxIter}")
        
        print(f"temp départ: {temp}")
        print(f"Ratio beta + / beta -: {ratio_temperature}")
        print("Règle de recuit : Décroissance exponentielle : temp_decay_factor = (temp_target / temp) ** (1.0/maxIter)")


        nl()

        # Unpack the two lists directly
        weights, temps, gammas, errors = minim_error_2_temp(L_ens, w, maxIter=maxIter, eta=eta, temp_init=temp, ratio_beta=ratio_temperature)
        
        viz = PerceptronVisualizer(L_ens, weights, 
                            (temps, "Temperature"), 
                            (gammas, "Mean Gamma"),
                            (errors, "Error")
                            )

        err = error(L_ens, weights[-1])
        print("last erreur d'apprentissage minimerror 2T : ", err)
        min_err = min(errors)
        best_iteration = errors.index(min_err)   

        print(f"Minimum error found by Minimerror 2T: {min_err} at iteration {best_iteration}")

        err = error(L_ens, w_prof)
        print("erreur d'apprentissage teacher : ", err)
    
        viz.save_tracks_separately(prefix="Q2_0/Q2_0 - ")
        viz.save_final_2d_plot("Q2_0/Q2_0 - final_result.png")
        viz.save_training_video("Q2_0/Q2_0 - perceptron_training.mp4", fps=15)

        df_weights = display_and_save_weights_table(
            weights[-1],
            filename="Q2_0/Q2_0_Poids_Perceptron.png",
            title="Q2_0 - Poids du Perceptron après Entraînement"
        )
    if if_question_2_1:
        print("-"*25, "QUESTION 2_1", "-"*25)
        print("Minimerror avec deux températures Beta+ / Beta- - Test sur un ensemble aléatoire garanti NON LS avec technique du Perceptron Professeur + Inversion")
        seed_1 = 45
        seed_2 = 99
        seed_3 = 21

        nbPoints = 200
        N = 2
        bornes = [-20,20]
        print("Points générés pour nbPoints =", nbPoints, ", N =", N, ", bornes =", bornes, ", seed =", seed)
        X_ens = create_points(nbPoints, N, bornes, seed_1)

        w_prof =init_perceptron_prof_in_bounds(X_ens, seed_2)

        L_ens = L_prof(X_ens, w_prof)

        facter_bruit = 20
        inversedExemple = int(nbPoints/facter_bruit)
        print("inversedExemple :", inversedExemple)
        L_ens = make_L_non_LS(L_ens, inversedExemple, seed_3)
        
        w = np.array(f_init_Hebb(L_ens, [1000, 1000]))
        w = w / np.linalg.norm(w)
        print("Initial weights w:", w)

        nl()

        maxIter = 1000
        eta = 0.001
        temp=0.25
        ratio_temperature = 1.0

        print(f"eta: {eta}")
        print(f"maxIter: {maxIter}")
        
        print(f"temp départ: {temp}")
        print(f"Ratio beta + / beta -: {ratio_temperature}")
        print("Règle de recuit : Décroissance exponentielle : temp_decay_factor = (temp_target / temp) ** (1.0/maxIter)")


        nl()

        # Unpack the two lists directly
        weights, temps, gammas, errors = minim_error_2_temp(L_ens, w, maxIter=maxIter, eta=eta, temp_init=temp, ratio_beta=ratio_temperature)
        
        viz = PerceptronVisualizer(L_ens, weights, 
                            (temps, "Temperature"), 
                            (gammas, "Mean Gamma"),
                            (errors, "Error")
                            )
        
        

        err = error(L_ens, weights[-1])
        print("last erreur d'apprentissage minimerror : ", err)
        min_err = min(errors)
        best_iteration = errors.index(min_err)   

        print(f"Minimum error found by Minimerror: {min_err} at iteration {best_iteration}")

        err = error(L_ens, w_prof)
        print("erreur d'apprentissage teacher : ", err)
    
        viz.save_tracks_separately(prefix="Q2_1/Q2_1 - ")
        viz.save_final_2d_plot("Q2_1/Q2_1 - final_result.png")
        viz.save_training_video("Q2_1/Q2_1 - perceptron_training.mp4", fps=15)

        df_weights = display_and_save_weights_table(
            weights[-1],
            filename="Q2_1/Q2_1_Poids_Perceptron.png",
            title="Q2_1 - Poids du Perceptron après Entraînement"
        )

    if if_question_2_2:
        print("-"*25, "QUESTION 2_2", "-"*25)
        print("Minimerror avec deux températures Beta+ / Beta- Entraînement sur ensemble L_Train et test sur ensemble L_Test")
        
        print("train_df_prepare")
        print(train_df_prepare)
        print("test_df_prepare")
        print(test_df_prepare)
        nl()

        w = np.array(f_init_Hebb(train_df_prepare, [1000, 1000]))
        w = w / np.linalg.norm(w)

        print("Initial weights w:", w)


        nl()

        maxIter = 4000
        eta = 0.0001
        temp=0.25
        ratio_temperature = 1.18

        print(f"eta: {eta}")
        print(f"maxIter: {maxIter}")
        
        print(f"temp départ: {temp}")
        print(f"Ratio beta + / beta -: {ratio_temperature} - Ratio de convergence optimale - Déterminé empiriquement en testant plusieurs valeurs")
        print("Règle de recuit : Décroissance exponentielle : temp_decay_factor = (temp_target / temp) ** (1.0/maxIter)")


        weights, temps, gammas, errors = minim_error_2_temp(train_df_prepare, w, maxIter=maxIter, eta=eta, temp_init=temp, ratio_beta=ratio_temperature)

        weight_norms = [np.linalg.norm(w) for w in weights]

    
        viz = PerceptronVisualizer(train_df_prepare, weights, 
                            (temps, "Temperature"), 
                            (gammas, "Mean Gamma"),
                            (errors, "Error"),
                            (weight_norms, "Weight Norm"),
                            show_plot=False)
        

        
        viz.save_tracks_separately(prefix="Q2_2/Q2_2 - ")

        stabilites_train = []
        stabilites_paires = stabilites(train_df_prepare, weights[-1])
        
        for paire in stabilites_paires:
            stabilites_train.append(paire[1])
        
        draw_curve_and_save(
            stabilites_train, 
            label_text='Stabilité', 
            plot_title="Stabilité gamma sur Ensemble d entrainement (L_train)", 
            filename='Q2_2/Q2_2 - Stabilité sur ensemble d entrainement',
            color='blue', 
            xlabel='Patrons', 
            ylabel='Stabilité gamma sur Ensemble d entrainement', 
            linewidth=2,
            scatter_or_plot='scatter'
        )

        err = error(train_df_prepare, weights[-1])
        print("last erreur d'apprentissage minimerror : ", err)
        min_err = min(errors)
        best_iteration = errors.index(min_err)   

        print(f"Minimum error found by Minimerror: {min_err} at iteration {best_iteration}")

        nl()
        print("-"*25, "TEST SUR test_df_prepare", "-"*25)
        E_g = error(test_df_prepare, weights[-1])
    

        print("Erreur de généralisation :", E_g)

        df_weights = display_and_save_weights_table(
            weights[-1],
            filename="Q2_2/Q2_2_Poids_Perceptron.png",
            title="Q2_2 - Poids du Perceptron après Entraînement"
        )

    if if_question_2_3:
        print("-"*25, "QUESTION 2_3", "-"*25)
        print("Minimerror avec deux températures Beta+ / Beta- Entraînement sur ensemble L_Train+L_Test")
        
        complete_df_prepare = pd.concat([train_df_prepare, test_df_prepare], ignore_index=True)
    
        w = np.array(f_init_Hebb(complete_df_prepare, [1000, 1000]))
        w = w / np.linalg.norm(w)

        print("Initial weights w:", w)


        nl()
        maxIter = 5000
        eta = 0.0001
        temp = 0.25
        ratio_temperature = 14
    
        print(f"eta: {eta}")
        print(f"maxIter: {maxIter}")
        
        print(f"temp départ: {temp}")
        print(f"Ratio beta + / beta -: {ratio_temperature} - Ratio de convergence optimale - Déterminé empiriquement en testant plusieurs valeurs")
        print("Règle de recuit : Décroissance exponentielle : temp_decay_factor = (temp_target / temp) ** (1.0/maxIter)")

        # Unpack the two lists directly
        weights, temps, gammas, errors = minim_error_2_temp(complete_df_prepare, w, 
                                                         maxIter=maxIter, eta=eta,
                                                         temp_init=temp, ratio_beta=ratio_temperature)

        weight_norms = [np.linalg.norm(w) for w in weights]

        viz = PerceptronVisualizer(complete_df_prepare, weights, 
                            (temps, "Temperature"), 
                            (gammas, "Mean Gamma"),
                            (errors, "Error"),
                            (weight_norms, "Weight Norm"),
                            show_plot=False)
        

        
        viz.save_tracks_separately(prefix="Q2_3/Q2_3 - Train+Test - ")

        stabilites_all = []
        stabilites_paires = stabilites(complete_df_prepare, weights[-1])
        
        for paire in stabilites_paires:
            stabilites_all.append(paire[1])
        
        draw_curve_and_save(
            stabilites_all, 
            label_text='Stabilité', 
            plot_title="Stabilité gamma sur Ensembles d entrainement + test (L_train + L_test)", 
            filename='Q2_3/Q2_3 - Stabilité sur ensembles d entrainement + test',
            color='blue', 
            xlabel='Patrons', 
            ylabel='Stabilité gamma sur Ensembles d entrainement + test', 
            linewidth=2,
            scatter_or_plot='scatter'
        )

        err = error(complete_df_prepare, weights[-1])
        print("last erreur d'apprentissage minimerror sur Ensembles d'entrainement + test : ", err)
        min_err = min(errors)
        best_iteration = errors.index(min_err)   

        print(f"Minimum error found by Minimerror: {min_err} at iteration {best_iteration}")

    print("\n" + "="*80)
    print("FIN DU TP3")
    print("="*80)


