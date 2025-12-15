import pandas as pd
import numpy as np
from random import uniform as rd
from typing import List, Dict, Any, Union
import matplotlib.pyplot as plt
import os
import copy
import dataframe_image as dfi

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

def n():
    print('-'*50, '\n')

def signe(x):
    if x > 0:
        return 1
    else:
        return -1

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
        for i in range(1, dim):
            w[i] = w[i] + L_ens[k][0][i] * L_ens[k][1]

    return w

def f_init_rand(L_ens, biais_range):
    if isinstance(L_ens, pd.DataFrame):
        print("[Avertissement] Conversion du DataFrame en Liste pour la fonction d'initialisation.")
        
        # Le DataFrame est converti en une liste de [features, target]
        L_ens_converted = []
        for index, row in L_ens.iterrows():
            L_ens_converted.append([row['Donnee'], row['Classe voulue']])
        L_ens = L_ens_converted
        
    p = len(L_ens)
    dim = len(L_ens[0][0]) #This dimension assumes features have already been biaised 

    borne_inf = biais_range[0]
    borne_sup = biais_range[1]

    w = []
    for i in range(dim):
        w.append(0.0)

    biais = (rd(borne_inf,borne_sup))
    w[0] = biais #Ajout du biais
    for i in range(1,dim):
        w[i] = (rd(borne_inf,borne_sup))

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

def project_data_on_3D(data):
    """Projection 3D avec PCA manuelle"""
    X = np.array(data['Donnee'].tolist())
    y = data['Classe voulue'].map({'M': 1, 'R': -1}).values

    # On exclut la colonne de biais (1.0) pour la standardisation et la PCA
    X_features = X[:, 1:]
    
    # Standardisation manuelle
    X_scaled, mean, std = standardize_features(X_features)

    # PCA manuelle
    X_3d, components, explained_variance = pca_manual(X_scaled, n_components=3)

    # Séparer les points selon les classes
    X_mines = X_3d[y == 1]
    X_rocks = X_3d[y == -1]
    
    # Setup de la figure et des axes 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d') 

    # Plotter les mines
    ax.scatter(X_mines[:, 0], X_mines[:, 1], X_mines[:, 2],
               label='Mine (M)', marker='o', color='red', alpha=0.7, s=50)

    # Plotter les roches
    ax.scatter(X_rocks[:, 0], X_rocks[:, 1], X_rocks[:, 2],
               label='Roche (R)', marker='x', color='blue', alpha=0.7, s=50)

    # Mise à jour des labels
    ax.set_xlabel(f'PC 1 ({explained_variance[0]*100:.2f}%)')
    ax.set_ylabel(f'PC 2 ({explained_variance[1]*100:.2f}%)')
    ax.set_zlabel(f'PC 3 ({explained_variance[2]*100:.2f}%)')
    
    ax.set_title('Projection 3D du Sonar Data (Mines vs Roches) via PCA')
    ax.legend()
    
    plt.savefig('sonar_pca_3d.png')
    plt.show()

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
    
def run_training_exo_1(data, training_algo, train, test, train_prepare, test_prepare, question_tag, if_show=True):
        biais_range = [-1, 1]
        perceptron = f_init_rand(train, biais_range)
        
        #Entrainement
        eta = 0.1
        maxIter = 10000
        trained_perceptron, _ ,training_errors = training_algo(perceptron, train_prepare, eta, maxIter) #La fonction retourne aussi le nombre d'itérations
        
        fig_2d, ax_2d = plt.subplots(figsize=(12, 10))
        draw_curve(ax_2d, training_errors, f"Erreur d'apprentissage - {question_tag}", color='red')
        fig_2d.savefig(f"Erreur d'apprentissage - {question_tag}.png")
        plt.close(fig_2d)

        #Calcul de l'erreur
        training_error = error(train_prepare, trained_perceptron)
        generalisation_error = error(test_prepare, trained_perceptron)
        
        print("Erreur d'entraînement : ", training_error)
        print('Error de généralisation : ', generalisation_error)

        #Calcul des stabilités
        stabilites_paires = stabilites(train_prepare, trained_perceptron)
        
        stabilites_list=[]
        for paire in stabilites_paires:
            stabilites_list.append(paire[1])
        

        return (trained_perceptron,
                training_error,
                generalisation_error,
                stabilites_paires,
                stabilites_list,
        )

def question_4(train_df_prepare, test_df_prepare, initialization, target_error, eta, maxIter=10000):   
                
    print(f"\n--- Initialisation: {initialization.__name__}, Delta (eta): {eta} ---")
    
    biais = [-1, 1]
    w_init =initialization(train_df_prepare, biais)

    # Apprentissage avec Pocket
    w_pocket, Ea, n_iter, error_history = pocket_algorithm(
        train_df_prepare, w_init, eta, maxIter, target_error
    )
    
    # Test sur test
    Eg = error(test_df_prepare, w_pocket)

    print(f"Ea: {Ea}/{len(train_df_prepare)} = {Ea/len(train_df_prepare)*100:.2f}%")
    print(f"Eg: {Eg}/{len(test_df_prepare)} = {Eg/len(test_df_prepare)*100:.2f}%")
    print(f"Itérations: {n_iter}")

    return initialization.__name__, eta, Ea, Eg, n_iter, error_history, w_pocket

def question_5(train_df_prepare, test_df_prepare, training_algo, maxIter=10000):
    # Créer l'ensemble complet L = train + test
    complete_df_prepare = pd.concat([train_df_prepare, test_df_prepare], ignore_index=True)
    print(f"\n{'='*80}")
    print("INFORMATIONS SUR LES DONNÉES")
    print(f"{'='*80}")
    print(f"Train: {len(train_df_prepare)} exemples")
    print(f"Test: {len(test_df_prepare)} exemples")
    print(f"Ensemble complet L: {len(complete_df_prepare)} exemples")
    print(f"Nombre de dimensions N: 60")
    print(f"Nombre de poids (N+1): 61")
    print(f"{'='*80}\n")
   
    print("Note: Utilisation de l'algorithme perceptron ONLINE (justification basée sur TP1)")
    print("     - Plus efficace en mémoire pour de grands ensembles")
    print("     - Mise à jour immédiate des poids après chaque exemple")
    print("     - Convergence généralement plus rapide pour ce type de problème\n")

    # Affichage des poids
    # n()
    # print("CHECKPOINT 1")


    # Paramètres d'apprentissage
    biais_range = [-1, 1]
    eta = 0.1
    # Initialisation et apprentissage
    w_init = f_init_rand(complete_df_prepare, biais_range)

    #Penser à plotter training_errors
    w_trained, n_iter, training_errors = training_algo(w_init, complete_df_prepare, eta, maxIter)
    
    fig_2d, ax_2d = plt.subplots(figsize=(12, 10))
    draw_curve(ax_2d, training_errors, "Erreur d'apprentissage - Online - Ensemble complet", color='red')
    fig_2d.savefig("Q5/Q5_Erreur d'apprentissage - Online - Ensemble complet.png")
    plt.close(fig_2d)

    # print("CHECKPOINT 0")
    # n()
    # Calcul des erreurs
    err = error(complete_df_prepare, w_trained)
    
    print_weights(w_trained, "Poids du perceptron")
    n()
    print(f"Erreur à l'itération {n_iter} : {err}/{len(complete_df_prepare)}")

    return w_trained, n_iter, err, complete_df_prepare
    
def question_6(train_df_prepare, test_df_prepare, eta, maxIter=10000):
        # Mélanger les données
    complete_df_prepare = pd.concat([train_df_prepare, test_df_prepare], ignore_index=True)
    L_shuffled = complete_df_prepare.sample(frac=1, random_state=exp).reset_index(drop=True)
    
    # Diviser: 50% LA, 25% LV, 25% LT
    n_total = len(L_shuffled)
    n_LA = int(0.5 * n_total)
    n_LV = int(0.25 * n_total)
    
    LA = L_shuffled[:n_LA].copy()
    LV = L_shuffled[n_LA:n_LA+n_LV].copy()
    LT = L_shuffled[n_LA+n_LV:].copy()
    
    print(f"LA: {len(LA)} exemples, LV: {len(LV)} exemples, LT: {len(LT)} exemples")
    
    # Initialisation
    biais = [-1, 1]
    w_init_ES = f_init_rand(LA, biais)
    
    # Early Stopping
    w_best_ES, best_Ev, n_iter_ES, train_errors_ES, val_errors_ES = early_stopping(
        LA, LV, w_init_ES, eta, maxIter, patience=20
    )
    
    # Calcul des erreurs
    Ea_ES = error(LA, w_best_ES)
    Ev_ES = best_Ev
    Et_ES = error(LT, w_best_ES)
 

    print(f"Ea: {Ea_ES}/{len(LA)} = {Ea_ES/len(LA)*100:.2f}%")
    print(f"Ev: {Ev_ES}/{len(LV)} = {Ev_ES/len(LV)*100:.2f}%")
    print(f"Et: {Et_ES}/{len(LT)} = {Et_ES/len(LT)*100:.2f}%")
    print(f"Itérations: {n_iter_ES}")

    return Ea_ES, Ev_ES, Et_ES, n_iter_ES, w_best_ES, train_errors_ES, val_errors_ES


f = False
t = True

if_question_2_et_3 = f
# if_question_2_et_3 = t

if_question_4 = f
# if_question_4 = t

# if_question_5 = f
if_question_5 = t

if_question_6 = f
# if_question_6 = t

if __name__=="__main__":

    #IMPORTATION DONNEES
    data_df, test_df, train_df = import_data()

    #PRETRAITEMENT
    train_df_prepare, test_df_prepare = pretraitement(test_df, train_df)

    if if_question_2_et_3:
        #QUESTION 2 et 3
        #L'algorithme online converge en moins d'itération que l'algorithme batch
        training_algo = perceptron_online
        # Question 2
        question_tag_exo_2 = 'Q2'
        trained_perceptron, training_error, generalisation_error, stabilites_paires, stabilites_list, = run_training_exo_1(data=data_df,
                                                                                                                        training_algo=training_algo, 
                                                                                                                        train=train_df, 
                                                                                                                        test=test_df, 
                                                                                                                        train_prepare=train_df_prepare,
                                                                                                                        test_prepare=test_df_prepare,
                                                                                                                        question_tag=question_tag_exo_2)
        
        df_weights = display_and_save_weights_table(
            trained_perceptron,
            filename="Q2_et_3/Q2_Poids_Perceptron.png",
            title="Q2 - Poids du Perceptron après Entraînement"
        )

        #Calcul des stabilités
        stabilites_train = stabilites_list #Retourné par la fonction d'entrainement
        draw_curve_and_save(
            stabilites_train, 
            label_text='Stabilité', 
            plot_title="Stabilité gamma sur Ensemble de entrainement", 
            filename='Q2_et_3/Q2 - Stabilité sur ensemble de entrainement',
            color='blue', 
            xlabel='Patrons', 
            ylabel='Stabilité gamma sur Ensemble de entrainement', 
            linewidth=2,
            scatter_or_plot='scatter'
        )        
        
        stabilites_test = []
        stabilites_paires = stabilites(test_df_prepare, trained_perceptron)
        
        for paire in stabilites_paires:
            stabilites_test.append(paire[1])
        
        draw_curve_and_save(
            stabilites_test, 
            label_text='Stabilité', 
            plot_title="Stabilité gamma sur Ensemble de test", 
            filename='Q2_et_3/Q2 - Stabilité sur ensemble de test',
            color='blue', 
            xlabel='Patrons', 
            ylabel='Stabilité gamma sur Ensemble de test', 
            linewidth=2,
            scatter_or_plot='scatter'
        )

        # Question 3
        question_tag_exo_3 = 'Q3'
        trained_perceptron, training_error, generalisation_error, stabilites_paires, stabilites_list, = run_training_exo_1(data=data_df,
                                                                                                                        training_algo=training_algo,  
                                                                                                                        train=test_df, 
                                                                                                                        test=train_df, 
                                                                                                                        train_prepare=test_df_prepare,
                                                                                                                        test_prepare=train_df_prepare,
                                                                                                                        question_tag=question_tag_exo_3)
        
        df_weights = display_and_save_weights_table(
            trained_perceptron,
            filename="Q3_Poids_Perceptron.png",
            title="Q3 - Poids du Perceptron après Entraînement"
        )
        
        #Calcul des stabilités
        stabilites_train = stabilites_list #Retourné par la fonction d'entrainement
        draw_curve_and_save(
            stabilites_train, 
            label_text='Stabilité', 
            plot_title="Stabilité gamma sur Ensemble de entrainement (L_test)", 
            filename='Q2_et_3/Q3 - Stabilité sur ensemble de entrainement',
            color='blue', 
            xlabel='Patrons', 
            ylabel='Stabilité gamma sur Ensemble de entrainement', 
            linewidth=2,
            scatter_or_plot='scatter'
        )        
        
        stabilites_test = []
        stabilites_paires = stabilites(train_df_prepare, trained_perceptron)
        
        for paire in stabilites_paires:
            stabilites_test.append(paire[1])
        
        draw_curve_and_save(
            stabilites_test, 
            label_text='Stabilité', 
            plot_title="Stabilité gamma sur Ensemble de test (L_train)", 
            filename='Q2_et_3/Q3 - Stabilité sur ensemble de test',
            color='blue', 
            xlabel='Patrons', 
            ylabel='Stabilité gamma sur Ensemble de test', 
            linewidth=2,
            scatter_or_plot='scatter'
        )
    
    # # ========================================================================
    # # PARTIE II: Algorithme Pocket
    # # ========================================================================

    if if_question_4:
        print(f"\n{'='*80}")
        print("PARTIE II: ALGORITHME POCKET")
        print(f"{'='*80}\n")
        # Test avec différentes initialisations et etas
        etas = [0.01, 0.1, 0.5, 1.0]
        initialization_test_case = [f_init_rand, f_init_Hebb]  
        
        # Arrêter quand Ea <= 5
        target_error = 5 

        results_pocket = []
        perceptrons_trained = []
        
        # Test en échangeant train et test
        print(f"\n{'='*60}")
        print("Pocket: Apprentissage sur 'train', test sur 'test'")
        print(f"{'='*60}\n")

        for initialization in initialization_test_case:
            for eta in etas:
                result = question_4(train_df_prepare, 
                                    test_df_prepare,
                                    initialization,
                                    target_error,
                                    eta)

                results_pocket.append({
                    'init': result[0],
                    'eta': result[1],
                    'Ea': result[2],
                    'Eg': result[3],
                    'iterations': result[4],
                    'Ens. Train': 'Train',
                    'Ens. Test': 'Test',
                })
                perceptrons_trained.append(result[6])
                
        # Test en échangeant train et test
        print(f"\n{'='*60}")
        print("Pocket: Apprentissage sur 'test', test sur 'train'")
        print(f"{'='*60}\n")
        
        for initialization in initialization_test_case:
            for eta in etas:
                result = question_4(test_df_prepare,
                                    train_df_prepare, 
                                    initialization,
                                    target_error,
                                    eta)

                results_pocket.append({
                    'init': result[0],
                    'eta': result[1],
                    'Ea': result[2],
                    'Eg': result[3],
                    'iterations': result[4],
                    'Ens. Train': 'Test',
                    'Ens. Test': 'Train',
                })
                perceptrons_trained.append(result[6])
                

        
        # Tableau récapitulatif
        print(f"\n{'='*80}")
        print("TABLEAU RÉCAPITULATIF - ALGORITHME POCKET")
        print(f"{'='*80}")
        print(f"{'Init':<10} {'Eta':<8} {'Ea':<8} {'Eg':<8} {'Itérations':<12} {'Ens. Train'} {'Ens. Test'}")
        print("-" * 80)
        for r in results_pocket:
            train_set = r.get('train_set', 'train')
            print(f"{r['init']:<10} {r['eta']:<8.2f} {r['Ea']:<8} {r['Eg']:<8} {r['iterations']:<12} {r['Ens. Train']:<12} {r['Ens. Test']}")
        print(f"{'='*80}\n")
        
        pocket_column_map = {
            'init':       'Initialisation',
            'eta':        'Eta (η)',                 
            'Ea':         'Erreur Appr. (Ea)',       
            'Eg':         'Erreur Gén. (Eg)',        
            'iterations': 'Itérations',
            'Ens. Train': 'Ensemble d\'Entraînement', 
            'Ens. Test':  'Ensemble de Test'         
        }
        display_and_save_results_table(results_pocket, 
                                       pocket_column_map,
                                       filename='Q4/Q4_pocket_results_summary.png',
                                       title='Résumé des résultats de l\'algorithme Pocket')
        
        display_and_save_matrix_table(perceptrons_trained,
                                      "Q4/Q4_Poids perceptrons entrainés Pocket.png",
                                      "Poids perceptrons entrainés Pocket",
                                      "Poid",
                                      "Perceptron test case")
    
        #Evaluation des stabilité de généralisation sur les 104 exemple
        #de l'ensemble de généralisation :

       # Matrice 1: Stabilités de généralisation sur l'ensemble de test (perceptrons 0-7)
        stabilites_generalisation_sur_test_matrix = []  # 104 exemples × 8 perceptrons

        # Matrice 2: Stabilités de généralisation sur l'ensemble de train (perceptrons 8-15)  
        stabilites_generalisation_sur_train_matrix = []  # 104 exemples × 8 perceptrons

        print("Construction des matrices de stabilités de généralisation...")

        # Pour les 8 premiers perceptrons (entraînés sur train, testés sur test)
        print(f"\n1. Perceptrons 1-8 (entraînés sur train, testés sur test)")
        for i in range(8):
            perceptron = perceptrons_trained[i]
            print(f"  Perceptron {i+1}/8...")
            
            stabilites_paires = stabilites(test_df_prepare, perceptron)
            print(f"    {len(stabilites_paires)} stabilités calculées")
            
            if i == 0:
                # Initialiser la matrice test (104 × 8)
                for paire in stabilites_paires:
                    stabilites_generalisation_sur_test_matrix.append([paire[1]])  # Ligne avec 1 valeur
            else:
                # Ajouter aux lignes existantes
                for idx, paire in enumerate(stabilites_paires):
                    stabilites_generalisation_sur_test_matrix[idx].append(paire[1])

        # Pour les 8 derniers perceptrons (entraînés sur test, testés sur train)
        print(f"\n2. Perceptrons 9-16 (entraînés sur test, testés sur train)")
        for i in range(8, 16):
            perceptron = perceptrons_trained[i]
            print(f"  Perceptron {i+1}/16...")
            
            stabilites_paires = stabilites(train_df_prepare, perceptron)
            print(f"    {len(stabilites_paires)} stabilités calculées")
            
            if i == 8:
                # Initialiser la matrice train (104 × 8)
                for paire in stabilites_paires:
                    stabilites_generalisation_sur_train_matrix.append([paire[1]])  # Ligne avec 1 valeur
            else:
                # Ajouter aux lignes existantes
                for idx, paire in enumerate(stabilites_paires):
                    stabilites_generalisation_sur_train_matrix[idx].append(paire[1])

        # Vérifications
        print(f"\n{'='*50}")
        print("RÉSUMÉ DES MATRICES DE STABILITÉS DE GÉNÉRALISATION")
        print(f"{'='*50}")
        print(f"Matrice sur TEST (perceptrons 1-8 sur exemples test):")
        print(f"  - Dimensions: {len(stabilites_generalisation_sur_test_matrix)} × {len(stabilites_generalisation_sur_test_matrix[0]) if stabilites_generalisation_sur_test_matrix else 0}")
        print(f"  - Attendue: 104 × 8")

        print(f"\nMatrice sur TRAIN (perceptrons 9-16 sur exemples train):")
        print(f"  - Dimensions: {len(stabilites_generalisation_sur_train_matrix)} × {len(stabilites_generalisation_sur_train_matrix[0]) if stabilites_generalisation_sur_train_matrix else 0}")
        print(f"  - Attendue: 104 × 8")

        # Afficher les matrices
        if stabilites_generalisation_sur_test_matrix:
            display_and_save_matrix_table(
                stabilites_generalisation_sur_test_matrix, 
                "Q4/Q4_Stabilites_generalisation_des_8_perceptrons_entraines_sur_train_testes_sur_test.png", 
                "Stabilités de généralisation des 8 perceptrons (entraînés sur train, testés sur test)",
                "Perceptron",  # Maintenant ce sera les lignes (après transposition)
                "Patron",      # Maintenant ce sera les colonnes (après transposition)
                batch_size=50  # Vous pouvez ajuster si nécessaire
            )

        if stabilites_generalisation_sur_train_matrix:
            display_and_save_matrix_table(
                stabilites_generalisation_sur_train_matrix, 
                "Q4/Q4_Stabilites_generalisation_des_8_perceptrons_entraines_sur_test_testes_sur_train.png", 
                "Stabilités de généralisation des 8 perceptrons (entraînés sur test, testés sur train)",
                "Perceptron",  # Maintenant ce sera les lignes (après transposition)
                "Patron",      # Maintenant ce sera les colonnes (après transposition)
                batch_size=50
    )
            
    # ========================================================================
    # PARTIE III: Test si L (train+test) est linéairement séparable
    # ========================================================================

    # Question 5

    if if_question_5:
        question_tag_exo_5 = 'EXERCICE 5'
        
        training_algo = perceptron_online
        maxIters = [10000, 20000, 30000, 50000, 100000, 200000, 500000]

        perceptrons_trained = []
        erreurs = []
        for i, maxIter in enumerate(maxIters) :
            print(f"{question_tag_exo_5} - Test case {i}")
            perceptron_entraine, _, err, complete_df_prepare = question_5(train_df_prepare,
                                                    test_df_prepare,
                                                    training_algo,
                                                    maxIter)
            perceptrons_trained.append(perceptron_entraine)
            erreurs.append(err)
            n()
        
        display_and_save_matrix_table(perceptrons_trained, "Q5\Q. 5 - Poids perceptrons sur l'ensemble entier.png", "Q. 5 - Poids perceptrons sur l'ensemble entier", "Poid", "Perceptron test case")

        stabilites_matrix = []  # 208 exemples × 3 perceptrons
        
        #Evaluation des stabilité (Par perceptron, sur chaque patron)
        for i in range(len(perceptrons_trained)):
            perceptron = perceptrons_trained[i]
            
            stabilites_paires = stabilites(complete_df_prepare, perceptron)
            print(f"    {len(stabilites_paires)} stabilités calculées")
            
            if i == 0:
                # Initialiser la matrice train (104 × 8)
                for paire in stabilites_paires:
                    stabilites_matrix.append([paire[1]])  # Ligne avec 1 valeur
            else:
                # Ajouter aux lignes existantes
                for idx, paire in enumerate(stabilites_paires):
                    stabilites_matrix[idx].append(paire[1])
        
        display_and_save_matrix_table(stabilites_matrix, "Q5\Q5_Stabilites des perceptrons entrainés sur l'ensemble complet.png", 
                                      "Stabilites des perceptrons entrainés sur l'ensemble complet", "Perceptron", "Patron")

        for i, err in enumerate(erreurs):
            print(f'Iterations {maxIters[i]} : Erreur = {err}/{len(complete_df_prepare)} = {err/len(complete_df_prepare)*100:.2f}%')        
        n()
        print("CONCLUSION")
        print("L'algortihme ne converge jamais : on en conclut que " \
            "L n'est pas linéairement séparable. En effet, P = 208 > 2N = 120" \
            "=> l'existence d'un hyperplan séparateur n'est pas garantie.")
    

    # # ========================================================================
    # # PARTIE IV: Early Stopping
    # # ========================================================================

    if if_question_6:
        print(f"\n{'='*80}")
        print("PARTIE IV: EARLY STOPPING")
        print(f"{'='*80}\n")
        
        # Répéter l'expérience plusieurs fois pour obtenir des statistiques
        n_experiments = 5
        eta = 0.1
        results_early_stopping = []
        
        perceptrons_trained=[]

        for exp in range(n_experiments):
            print(f"\n--- Expérience {exp+1}/{n_experiments} ---")
            
 
            result = question_6(train_df_prepare, test_df_prepare, eta)

            results_early_stopping.append({
                'Ea': result[0],
                'Ev': result[1],
                'Et': result[2],
                'iterations': result[3]
            })
            perceptrons_trained.append(result[4])
            
        #Plot de l'évolution des erreurs de la dernière expérience
        train_errors_ES = result[5]
        val_errors_ES = result[6]
        complete_df_prepare = pd.concat([train_df_prepare, test_df_prepare], ignore_index=True)
    
        display_and_save_matrix_table(perceptrons_trained, "Q6\Q. 6 - Poids perceptrons pour early stopping.png", "Q. 6 - Poids perceptrons pour early stopping", "Poid", "Perceptron test case")
        
        #Evaluation des stabilité (Par perceptron, sur chaque patron)
        stabilites_matrix = []
        
        for i in range(len(perceptrons_trained)):
            perceptron = perceptrons_trained[i]
            
            stabilites_paires = stabilites(complete_df_prepare, perceptron)
            print(f"    {len(stabilites_paires)} stabilités calculées")
            
            if i == 0:
                # Initialiser la matrice train (104 × 8)
                for paire in stabilites_paires:
                    stabilites_matrix.append([paire[1]])  # Ligne avec 1 valeur
            else:
                # Ajouter aux lignes existantes
                for idx, paire in enumerate(stabilites_paires):
                    stabilites_matrix[idx].append(paire[1])

        display_and_save_matrix_table(stabilites_matrix, "Q6\Q. 6 - Stabilites des perceptrons entrainés avec Early Stopping.png", 
                                      "Q. 6 - Stabilites des perceptrons entrainés avec Early Stopping", "Stabilité perceptron", "Patron")


        # Statistiques
        print(f"\n{'='*80}")
        print("STATISTIQUES - EARLY STOPPING (moyenne sur {} expériences)".format(n_experiments))
        print(f"{'='*80}")
        
        mean_Ea = np.mean([r['Ea'] for r in results_early_stopping])
        mean_Ev = np.mean([r['Ev'] for r in results_early_stopping])
        mean_Et = np.mean([r['Et'] for r in results_early_stopping])
        std_Ea = np.std([r['Ea'] for r in results_early_stopping])
        std_Ev = np.std([r['Ev'] for r in results_early_stopping])
        std_Et = np.std([r['Et'] for r in results_early_stopping])
        
        print(f"Ea (moyenne): {mean_Ea:.2f} ± {std_Ea:.2f}")
        print(f"Ev (moyenne): {mean_Ev:.2f} ± {std_Ev:.2f}")
        print(f"Et (moyenne): {mean_Et:.2f} ± {std_Et:.2f}")
        print(f"{'='*80}\n")
        
        # Graphique de l'évolution des erreurs (dernière expérience)
        plt.figure(figsize=(12, 6))
        plt.plot(train_errors_ES, label='Erreur d\'apprentissage (LA)', marker='o', markersize=3)
        plt.plot(val_errors_ES, label='Erreur de validation (LV)', marker='s', markersize=3)
        plt.xlabel('Itérations')
        plt.ylabel('Erreur')
        plt.title("Early Stopping - Évolution en fonction des itérations des erreurs d'entraînement et de validation")
        plt.legend()
        plt.grid(True)
        plt.savefig("Q6/Q6_Comparaison_err_entraînement_et_validation", bbox_inches='tight', dpi=300)            
        
        early_stopping_column_map = {
            # Clé dans results_early_stopping : Nom d'affichage dans le tableau
            'Ea':           'Erreur d\'Apprentissage (Ea)',
            'Ev':           'Erreur de Validation (Ev)',
            'Et':           'Erreur de Test (Et)',  # Ou 'Erreur de Généralisation (Et)' si vous préférez
            'iterations':   'Itérations'
        }
        display_and_save_results_table(results_early_stopping, 
                                       early_stopping_column_map,
                                       filename='Q6\Q6_results_early_stopping_summary.png',
                                       title='Résumé des résultats des test avec Early Stopping')
    


    print("\n" + "="*80)
    print("FIN DU TP2")
    print("="*80)


