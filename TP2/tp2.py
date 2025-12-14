import pandas as pd
import numpy as np
from random import uniform as rd
from typing import List, Dict, Any, Union
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Patch

from mpl_toolkits.mplot3d import Axes3D
import copy
import dataframe_image as dfi


def display_and_save_matrix_table(
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
        print(f"\n✅ Tableau sauvegardé avec succès dans le fichier : **{filename}**")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exportation en PNG : {e}")

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
        print(f"\n✅ Tableau sauvegardé avec succès dans le fichier : **{filename}**")
        
    except ImportError:
        print("\n⚠️ Erreur : La bibliothèque 'dataframe-image' n'est pas installée.")
        print("Veuillez l'installer avec : `pip install dataframe-image`")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exportation en PNG : {e}")
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
        print(f"\n✅ Tableau sauvegardé avec succès dans le fichier : **{filename}**")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exportation en PNG. Assurez-vous que 'dataframe-image' est installé.")
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

def error_J_wrong (L, w):
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
        
        err += (y[i] - signe(np.dot(w,X[i])))**2

    return err

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
    
    print(f"✅ Courbe sauvegardée avec succès dans le fichier : **{filename}**")
    
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

def standardize_features(X):
    """Standardisation manuelle des caractéristiques"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1.0, std)  # Éviter division par zéro
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def pca_project_sklearn(X, n_components=3, pca_model=None, scaler_mean=None, scaler_std=None, standardize=True):
    """
    Fonction PCA polyvalente utilisant scikit-learn.
    
    Args:
        X (np.ndarray): données (n_samples, n_features)
        n_components (int): nombre de composantes principales
        pca_model (PCA): modèle PCA pré-entraîné
        scaler_mean (np.ndarray): moyenne pour standardisation
        scaler_std (np.ndarray): écart-type pour standardisation
        standardize (bool): si True, standardiser les données
        
    Returns:
        X_projected (np.ndarray): données projetées
        pca_model (PCA): modèle PCA
        explained_variance (np.ndarray): variance expliquée par composante
        scaler_mean (np.ndarray): moyenne utilisée
        scaler_std (np.ndarray): écart-type utilisé
    """
    X = np.asarray(X, dtype=float)
    
    # Standardisation
    if standardize:
        if scaler_mean is None or scaler_std is None:
            scaler_mean = np.mean(X, axis=0)
            scaler_std = np.std(X, axis=0)
            # CORRECTION ICI : utiliser 'scaler_std' au lieu de 'std'
            scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)  # <-- CORRIGÉ
        X_scaled = (X - scaler_mean) / scaler_std
    else:
        if scaler_mean is None:
            scaler_mean = np.mean(X, axis=0)
        X_scaled = X - scaler_mean
        if scaler_std is None:
            scaler_std = np.ones(X.shape[1])
    
    # PCA - Fit ou Transform
    if pca_model is None:
        pca_model = PCA(n_components=n_components)
        X_projected = pca_model.fit_transform(X_scaled)
    else:
        X_projected = pca_model.transform(X_scaled)
    
    explained_variance = pca_model.explained_variance_ratio_
    
    return X_projected, pca_model, explained_variance, scaler_mean, scaler_std

def project_data_on_2D(data, pca_model=None, scaler_mean=None, scaler_std=None):
    """
    Projection 2D avec PCA.
    
    Args:
        data: DataFrame avec 'Donnee' et 'Classe voulue'
        pca_model: modèle PCA pré-entraîné (optionnel)
        scaler_mean: moyenne de standardisation (optionnel)
        scaler_std: écart-type de standardisation (optionnel)
        
    Returns:
        pca_model, scaler_mean, scaler_std (pour réutilisation)
    """
    # Préparation des données
    X = np.array(data['Donnee'].tolist())
    y = data['Classe voulue'].map({'M': 1, 'R': -1}).values
    X_features = X[:, 1:]  # Exclure le biais
    
    # PCA avec la fonction unifiée
    X_2d, pca_model, explained_variance, scaler_mean, scaler_std = pca_project_sklearn(
        X_features, 
        n_components=2,
        pca_model=pca_model,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        standardize=True
    )
    
    # Séparation par classe
    X_mines = X_2d[y == 1]
    X_rocks = X_2d[y == -1]
    
    # Visualisation
    plt.figure(figsize=(10, 8))
    plt.scatter(X_mines[:, 0], X_mines[:, 1], 
                label='Mine (M)', marker='o', color='red', alpha=0.6)
    plt.scatter(X_rocks[:, 0], X_rocks[:, 1], 
                label='Roche (R)', marker='x', color='blue', alpha=0.6)
    
    plt.xlabel(f'PC 1 ({explained_variance[0]*100:.2f}%)')
    plt.ylabel(f'PC 2 ({explained_variance[1]*100:.2f}%)')
    plt.title('Projection 2D du Sonar Data (Mines vs Roches) via PCA')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return pca_model, scaler_mean, scaler_std

def draw_data_points_3D(ax, data_df, pca_model=None, scaler_mean=None, scaler_std=None):
    """
    Projette les données dans l'espace 3D PCA et trace les points.
    
    Returns:
        pca_model, scaler_mean, scaler_std, X_3d
    """
    # Préparation des données
    X = np.array(data_df['Donnee'].tolist())
    y = data_df['Classe voulue'].map({'M': 1, 'R': -1}).values
    X_features = X[:, 1:]  # Exclure le biais
    
    # Utilisation de la fonction PCA unifiée
    X_3d, pca_model, explained_variance, scaler_mean, scaler_std = pca_project_sklearn(
        X_features,
        n_components=3,
        pca_model=pca_model,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        standardize=True
    )
    
    # Séparation par classe
    X_mines = X_3d[y == 1]
    X_rocks = X_3d[y == -1]
    
    # Tracé 3D
    ax.scatter(X_mines[:, 0], X_mines[:, 1], X_mines[:, 2],
               label='Mine (M)', marker='o', color='red', alpha=0.7, s=50)
    ax.scatter(X_rocks[:, 0], X_rocks[:, 1], X_rocks[:, 2],
               label='Roche (R)', marker='x', color='blue', alpha=0.7, s=50)
    
    # Labels des axes
    ax.set_xlabel(f'PC 1 ({explained_variance[0]*100:.2f}%)')
    ax.set_ylabel(f'PC 2 ({explained_variance[1]*100:.2f}%)')
    ax.set_zlabel(f'PC 3 ({explained_variance[2]*100:.2f}%)')
    
    # Transpose components to shape (n_features_original, n_components)
    return pca_model.components_.T, scaler_mean, scaler_std, X_3d

def draw_perceptron_plane_3D(ax, X_3d, perceptron, pca_components, scaler_mean, scaler_std, color='red', label='Hyperplan'):
    """
    Trace l'hyperplan du perceptron dans l'espace 3D PCA.
    
    Args:
        ax: axe 3D
        X_3d: données projetées en 3D (n_samples, 3)
        perceptron: modèle de perceptron
        pca_components: matrice de composantes PCA (n_features_original, 3)
        scaler_mean: moyenne de standardisation (n_features_original,)
        scaler_std: écart-type de standardisation (n_features_original,)
    """
    # Vérifier les dimensions
    if pca_components is None or scaler_mean is None or scaler_std is None:
        print("Attention: PCA components ou scaler non fournis, impossible de tracer l'hyperplan")
        return
    
    # 1. Extraire les poids du perceptron
    w = perceptron  # Shape: (n_features + 1, )
    w_bias = w[0]     # Biais
    w_features = w[1:]  # Poids des features, shape: (n_features_original,)
    
    # 3. Standardiser les poids (important!)
    w_features_scaled = w_features / scaler_std
    
    # 4. Projection dans l'espace PCA
    w_pca = w_features_scaled @ pca_components  # Shape: (3,)
    
    # 5. Construire le plan: w_pca·x + w_bias = 0
    #    => w1*x + w2*y + w3*z + w_bias = 0
    #    => z = (-w1*x - w2*y - w_bias) / w3
    
    # Générer une grille
    x_min, x_max = X_3d[:, 0].min() - 1, X_3d[:, 0].max() + 1
    y_min, y_max = X_3d[:, 1].min() - 1, X_3d[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20)
    )
    
    # Calculer z pour chaque point (x, y)
    # w_pca[0]*x + w_pca[1]*y + w_pca[2]*z + w_bias = 0
    # => z = (-w_pca[0]*x - w_pca[1]*y - w_bias) / w_pca[2]
    
    # Éviter la division par zéro
    if abs(w_pca[2]) > 1e-10:
        zz = (-w_pca[0] * xx - w_pca[1] * yy - w_bias) / w_pca[2]
    else:
        # Si w_pca[2] est proche de 0, le plan est presque vertical
        # On peut afficher un plan à z constant
        zz = np.zeros_like(xx)
        print("Attention: Composante z du poids PCA proche de 0, plan vertical")
    
    # Tracer le plan
    ax.plot_surface(xx, yy, zz, alpha=0.3, color=color, label=label)
    
    # Ajouter une légende
    ax.text(x_min, y_max, zz.max(), label, color=color, fontsize=10)

def visualize_perceptron_3D(data_df, w_initial, w_trained, title="Hyperplan séparateur du Perceptron"):
    """
    Visualise les données en 3D avec l'hyperplan séparateur du perceptron.
    Version sans sklearn.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dessiner les points de données
    pca_components, scaler_mean, scaler_std, X_3d = draw_data_points_3D(ax, data_df)
    
    # Dessiner l'hyperplan initial (optionnel, en rouge clair)
    if w_initial is not None:
        draw_perceptron_plane_3D(ax, X_3d, w_initial, pca_components, scaler_mean, scaler_std,
                               color='orange', alpha=0.2, label='Hyperplan initial')
    
    # Dessiner l'hyperplan entraîné (en vert, plus visible)
    if w_trained is not None:
        draw_perceptron_plane_3D(ax, X_3d, w_trained, pca_components, scaler_mean, scaler_std,
                               color='green', alpha=0.5, label='Hyperplan séparateur (entraîné)')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Créer une légende personnalisée
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='Mine (M)', alpha=0.7),
        plt.Line2D([0], [0], marker='x', color='blue', 
                   markersize=10, label='Roche (R)', alpha=0.7),
        Patch(facecolor='green', alpha=0.5, label='Hyperplan séparateur'),
    ]
    if w_initial is not None:
        legend_elements.append(Patch(facecolor='orange', alpha=0.2, label='Hyperplan initial'))
    
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    return fig, ax
    
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
        
        xlabel="Exemple p"
        ylabel="Stabilite p"
        fig_stabilites, ax_stabilites = plt.subplots(figsize=(12, 10))
        draw_scatter_plot(ax_stabilites, stabilites_list, 
                        f"Stabilites des exemples d'apprentissage - {question_tag}", 
                        color='red',
                        xlabel=xlabel,
                        ylabel=ylabel)
        #Empty : True - f : False
        ax_stabilites.legend()
        fig_stabilites.savefig(f"Stabilites des exemples d'apprentissage - {question_tag}.png")
        
        if if_show:
            plt.show()
        else:
            plt.close(fig_3d)  # Close 3D figure
            plt.close(fig_stabilites)

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

if __name__=="__main__":

    #IMPORTATION DONNEES
    data_df, test_df, train_df = import_data()

    #PRETRAITEMENT
    train_df_prepare, test_df_prepare = pretraitement(test_df, train_df)

    #ENTRAINTEMENT
    if_question_2_et_3 = True
    # if_question_2_et_3 = f
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
            filename="Q2_Poids_Perceptron.png",
            title="Q2 - Poids du Perceptron après Entraînement"
        )

        #Calcul des stabilités
        stabilites_train = stabilites_list #Retourné par la fonction d'entrainement
        draw_curve_and_save(
            stabilites_train, 
            label_text='Stabilité', 
            plot_title="Stabilité gamma sur Ensemble de entrainement", 
            filename='Q2 - Stabilité sur ensemble de entrainement',
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
            filename='Q2 - Stabilité sur ensemble de test',
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
            plot_title="Stabilité gamma sur Ensemble de entrainement", 
            filename='Q3 - Stabilité sur ensemble de entrainement',
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
            filename='Q3 - Stabilité sur ensemble de test',
            color='blue', 
            xlabel='Patrons', 
            ylabel='Stabilité gamma sur Ensemble de test', 
            linewidth=2,
            scatter_or_plot='scatter'
        )
    
    # # ========================================================================
    # # PARTIE II: Algorithme Pocket
    # # ========================================================================
    if_question_4 = True

    # if_question_4 = f
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
        display_and_save_results_table(results_pocket, pocket_column_map, filename='pocket_results_summary.png', title='Résumé des résultats de l\'algorithme Pocket')
        
        display_and_save_matrix_table(perceptrons_trained, "Poids perceptrons entrainés Pocket.png", "Poids perceptrons entrainés Pocket", "Poid", "Perceptron test case")
    
        #Evaluation des stabilité (Par perceptron, sur chaque patron)
        stabilites_matrix = []
        for i, perceptron in enumerate(perceptrons_trained):
            if i < 8: 
                stabilites_paires = stabilites(test_df_prepare, perceptron)
            else: #Les deuxième 8 perceptrons ont été entrainés sur test
                stabilites_paires = stabilites(train_df_prepare, perceptron)
            
            for paire in stabilites_paires:
                stabilites_matrix.append(paire[1])

        display_and_save_matrix_table(stabilites_matrix, "Stabilites des 16 perceptrons entrainés avc Pocket.png", 
                                      "Stabilites des 16 perceptrons entrainés avc Pocket", "Patron", "Stabilité perceptron")
    
            
    # ========================================================================
    # PARTIE III: Test si L (train+test) est linéairement séparable
    # ========================================================================

    # Question 5
    if_question_5 = True
    # if_question_5 = f
    if if_question_5:
        question_tag_exo_5 = 'EXERCICE 5'
        
        training_algo = perceptron_online
        maxIters = [10000, 20000, 30000]

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
        
        display_and_save_matrix_table(perceptrons_trained, "Q. 5 - Poids perceptrons sur l'ensemble entier", "Q. 5 - Poids perceptrons sur l'ensemble entier", "Poid", "Perceptron test case")

        #Evaluation des stabilité (Par perceptron, sur chaque patron)
        stabilites_matrix = []
        for i, perceptron in enumerate(perceptrons_trained):
            stabilites_paires = stabilites(complete_df_prepare, perceptron)
            stabilites_matrix.append(paire[1])

        display_and_save_matrix_table(stabilites_matrix, "Stabilites des perceptrons entrainés sur l'ensemble complet.png", 
                                      "Stabilites des perceptrons entrainés sur l'ensemble complet", "Patron", "Stabilité perceptron")

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
    if_question_6 = True
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

        display_and_save_matrix_table(perceptrons_trained, "Q. 6 - Poids perceptrons pour early stopping", "Q. 6 - Poids perceptrons pour early stopping", "Poid", "Perceptron test case")
        
        #Evaluation des stabilité (Par perceptron, sur chaque patron)
        stabilites_matrix = []
        for i, perceptron in enumerate(perceptrons_trained):
            stabilites_paires = stabilites(complete_df_prepare, perceptron)
            stabilites_matrix.append(paire[1])

        display_and_save_matrix_table(stabilites_matrix, "Q. 6 - Stabilites des perceptrons entrainés avec Early Stopping.png", 
                                      "Q. 6 - Stabilites des perceptrons entrainés avec Early Stopping", "Patron", "Stabilité perceptron")


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
        plt.title('Early Stopping - Évolution des erreurs')
        plt.legend()
        plt.grid(True)
        plt.show()
                    
        early_stopping_column_map = {
            # Clé dans results_early_stopping : Nom d'affichage dans le tableau
            'Ea':           'Erreur d\'Apprentissage (Ea)',
            'Ev':           'Erreur de Validation (Ev)',
            'Et':           'Erreur de Test (Et)',  # Ou 'Erreur de Généralisation (Et)' si vous préférez
            'iterations':   'Itérations'
        }
        display_and_save_results_table(results_early_stopping, 
                                       early_stopping_column_map,
                                       filename='results_early_stopping_summary.png',
                                       title='Résumé des résultats des test avec Early Stopping')
    


    print("\n" + "="*80)
    print("FIN DU TP2")
    print("="*80)


