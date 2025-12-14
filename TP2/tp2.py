import pandas as pd
import numpy as np
from random import uniform as rd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import copy

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

def perceptron_online(w_vect, L_ens, eta, maxIter):
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
    err.append(error(L_ens, w_vect))
    while stop < maxIter:     # boucle sur les époques
        # print('\n'*2, '_'*20, '\n')
        #print("stop =", stop)
        nb_mal_classe = 0
        err.append(error(L_ens, w_k))
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

def calculate_stability(L, w):
    """Calcule les stabilités (gamma) des exemples selon la formule:
    gamma = distance à l'hyperplan séparateur avec les poids normés
    gamma = (w^T * x) / ||w|| où w est normalisé (sans le biais pour la norme)
    
    Args:
        L: ensemble de données (DataFrame ou liste)
        w: vecteur de poids du perceptron (N+1 dimensions avec biais)
    
    Returns:
        list: liste des stabilités pour chaque exemple
    """
    if isinstance(L, pd.DataFrame):
        L_converted = []
        for index, row in L.iterrows():
            L_converted.append([row['Donnee'], row['Classe voulue']])
        L = L_converted
    
    stabilities = []
    w_array = np.array(w)
    
    # Calculer la norme de w (sans le biais, indices 1 à N)
    w_features = w_array[1:]
    w_norm = np.linalg.norm(w_features)
    
    if w_norm < 1e-10:
        print("[AVERTISSEMENT] Norme des poids trop petite pour calculer les stabilités")
        return [0.0] * len(L)
    
    for exemple in L:
        x = np.array(exemple[0])
        # Produit scalaire w^T * x
        w_dot_x = np.dot(w_array, x)
        # Stabilité = distance signée à l'hyperplan
        gamma = w_dot_x / w_norm
        stabilities.append(gamma)
    
    return stabilities

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
    """Standardisation manuelle: centrer et réduire"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Éviter division par zéro
    return (X - mean) / std, mean, std

def pca_manual(X, n_components=3):
    """PCA manuelle sans sklearn"""
    # Centrer les données
    X_centered = X - np.mean(X, axis=0)
    
    # Calculer la matrice de covariance
    cov_matrix = np.cov(X_centered.T)
    
    # Calculer les valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Trier par ordre décroissant des valeurs propres
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Prendre les n_components premiers
    components = eigenvectors[:, :n_components]
    
    # Projection
    X_projected = X_centered @ components
    
    # Variance expliquée
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_projected, components, explained_variance

def project_data_on_2D(data):
    """Projection 2D avec PCA manuelle"""
    X = np.array(data['Donnee'].tolist())
    y = data['Classe voulue'].map({'M': 1, 'R': -1}).values

    X_features = X[:, 1:]  # Exclure le biais
    
    # Standardisation
    X_scaled, mean, std = standardize_features(X_features)
    
    # PCA
    X_2d, components, explained_variance = pca_manual(X_scaled, n_components=2)

    X_mines = X_2d[y == 1]
    X_rocks = X_2d[y == -1]

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

def draw_data_points_3D(ax, data_df, pca_components=None, scaler_mean=None, scaler_std=None):
    """
    Projette les données dans l'espace 3D PCA et trace les points sur l'axe donné.
    Version sans sklearn.
    
    Returns: pca_components, scaler_mean, scaler_std
    """
    
    # Préparation des données
    X = np.array(data_df['Donnee'].tolist())
    y = data_df['Classe voulue'].map({'M': 1, 'R': -1}).values
    X_features = X[:, 1:]  # Exclure le biais
    
    # 1. Standardisation et PCA
    if scaler_mean is None or scaler_std is None:
        X_scaled, mean, std = standardize_features(X_features)
        scaler_mean = mean
        scaler_std = std
    else:
        X_scaled = (X_features - scaler_mean) / scaler_std
        
    if pca_components is None:
        X_3d, components, explained_variance = pca_manual(X_scaled, n_components=3)
        pca_components = components
        pca_explained_variance = explained_variance
    else:
        X_centered = X_scaled - np.mean(X_scaled, axis=0)
        X_3d = X_centered @ pca_components
        # Pour l'affichage, on recalcule la variance expliquée
        _, _, pca_explained_variance = pca_manual(X_scaled, n_components=3)

    # Séparer les points selon les classes
    X_mines = X_3d[y == 1]
    X_rocks = X_3d[y == -1]

    # Tracé des points
    ax.scatter(X_mines[:, 0], X_mines[:, 1], X_mines[:, 2],
               label='Mine (M)', marker='o', color='red', alpha=0.7, s=50)
    ax.scatter(X_rocks[:, 0], X_rocks[:, 1], X_rocks[:, 2],
               label='Roche (R)', marker='x', color='blue', alpha=0.7, s=50)
               
    # Mise à jour des labels des axes
    ax.set_xlabel(f'PC 1 ({pca_explained_variance[0]*100:.2f}%)')
    ax.set_ylabel(f'PC 2 ({pca_explained_variance[1]*100:.2f}%)')
    ax.set_zlabel(f'PC 3 ({pca_explained_variance[2]*100:.2f}%)')
    
    return pca_components, scaler_mean, scaler_std, X_3d

def draw_perceptron_plane_3D(ax, X_3d, W_trained, pca_components, scaler_mean, scaler_std, 
                             color='green', alpha=0.4, label='Hyperplan séparateur'):
    """
    Trace le plan de séparation Perceptron en 3D.
    Version sans sklearn.
    
    Args:
        ax (Axes3D): L'objet Axes3D existant.
        X_3d: Données projetées en 3D (pour déterminer les limites)
        W_trained: Vecteur de poids entraîné (61 dimensions avec biais)
        pca_components: Composantes principales de la PCA
        scaler_mean: Moyenne du scaler
        scaler_std: Écart-type du scaler
        color: Couleur du plan
        alpha: Transparence du plan
        label: Label pour la légende
    """
    W_trained = np.asarray(W_trained, dtype=np.float64)
    
    # Extraire le biais et les poids des features
    w0 = W_trained[0]  # biais
    w_features = W_trained[1:]  # poids des 60 features
    
    # Standardiser les poids des features (comme les données)
    w_features_scaled = (w_features - scaler_mean) / scaler_std
    
    # Projection du vecteur de poids dans l'espace PCA
    # Le vecteur de poids dans l'espace original est projeté via les composantes PCA
    w_pca = w_features_scaled @ pca_components  # Shape: (3,)
    
    # L'équation du plan dans l'espace PCA est: w0 + w_pca[0]*x + w_pca[1]*y + w_pca[2]*z = 0
    # Donc: z = -(w0 + w_pca[0]*x + w_pca[1]*y) / w_pca[2]
    
    # Définition de la grille pour le plan
    x_min, x_max = X_3d[:, 0].min() - 1, X_3d[:, 0].max() + 1
    y_min, y_max = X_3d[:, 1].min() - 1, X_3d[:, 1].max() + 1
    
    PC1_grid, PC2_grid = np.meshgrid(np.linspace(x_min, x_max, 50),
                                     np.linspace(y_min, y_max, 50))
    
    # Calcul de PC3 (Z) : z = -(w0 + w_pca[0]*x + w_pca[1]*y) / w_pca[2]
    if abs(w_pca[2]) < 1e-10:
        print("[AVERTISSEMENT] Le poids PC3 est trop petit pour tracer le plan.")
        # Essayer une autre orientation
        if abs(w_pca[1]) > 1e-10:
            # Utiliser y au lieu de z
            PC2_grid, PC3_grid = np.meshgrid(np.linspace(x_min, x_max, 50),
                                           np.linspace(X_3d[:, 2].min() - 1, X_3d[:, 2].max() + 1, 50))
            PC1_grid = (-w0 - w_pca[1] * PC2_grid - w_pca[2] * PC3_grid) / w_pca[0]
            ax.plot_surface(PC1_grid, PC2_grid, PC3_grid, alpha=alpha, color=color)
        return

    PC3_grid = -(w0 + w_pca[0] * PC1_grid + w_pca[1] * PC2_grid) / w_pca[2]

    # Tracé du Plan de Séparation avec une meilleure visibilité
    surf = ax.plot_surface(PC1_grid, PC2_grid, PC3_grid, 
                          alpha=alpha, color=color, shade=True, 
                          linewidth=0, antialiased=True, label=label)
    
    # Ajouter des contours pour mieux voir le plan
    ax.contour(PC1_grid, PC2_grid, PC3_grid, zdir='z', offset=X_3d[:, 2].min() - 1, 
               colors=color, alpha=0.3, linewidths=1)

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

        #Affichage projection 3D pour visualisation de la convergence
        fig_3d = plt.figure(figsize=(12, 10))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        pca_components, scaler_mean, scaler_std, X_3d = draw_data_points_3D(ax_3d, train)
        draw_perceptron_plane_3D(ax_3d, X_3d, perceptron, pca_components, scaler_mean, scaler_std, color='red', label='Perceptron état initial')
        draw_perceptron_plane_3D(ax_3d, X_3d, trained_perceptron, pca_components, scaler_mean, scaler_std, color='blue', label='Perceptron état final')
        ax_3d.set_title(f'Etat initial et Séparation du Perceptron dans l\'Espace PCA 3D - {question_tag}')
        ax_3d.legend()
        fig_3d.savefig(f'Etat initial et Séparation du Perceptron dans l\'Espace PCA 3D - {question_tag}.png')
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
    #Variable générale
f = False

if __name__=="__main__":

    #IMPORTATION DONNEES
    data_df, test_df, train_df = import_data()

    #PRETRAITEMENT
    train_df_prepare, test_df_prepare = pretraitement(test_df, train_df)

    #ENTRAINTEMENT
    #L'algorithme online converge en moins d'itération que l'algorithme batch
    training_algo = perceptron_online
    # Question 2_a
    question_tag_exo_2 = 'Q2'
    trained_perceptron, training_error, generalisation_error, stabilites_paires, stabilites_list, = run_training_exo_1(data=data_df,
                                                                                                                      training_algo=training_algo, 
                                                                                                                      train=train_df, 
                                                                                                                      test=test_df, 
                                                                                                                      train_prepare=train_df_prepare,
                                                                                                                      test_prepare=test_df_prepare,
                                                                                                                      question_tag=question_tag_exo_2)
    # Question 2_a
    question_tag_exo_3 = 'Q3'
    trained_perceptron, training_error, generalisation_error, stabilites_paires, stabilites_list, = run_training_exo_1(data=data_df,
                                                                                                                      training_algo=training_algo,  
                                                                                                                      train=test_df, 
                                                                                                                      test=train_df, 
                                                                                                                      train_prepare=test_df_prepare,
                                                                                                                      test_prepare=train_df_prepare,
                                                                                                                      question_tag=question_tag_exo_3)

    #Initialisation d'un perceptron

# Créer l'ensemble complet L = train + test
    complete_df_prepare = pd.concat([train_df_prepare, test_df_prepare], ignore_index=True)
    
    print(f"\n{'='*80}")
    print("INFORMATIONS SUR LES DONNÉES")
    print(f"{'='*80}")
    print(f"Nombre total d'exemples: {len(data_df)}")
    print(f"Train: {len(train_df_prepare)} exemples")
    print(f"Test: {len(test_df_prepare)} exemples")
    print(f"Ensemble complet L: {len(complete_df_prepare)} exemples")
    print(f"Nombre de dimensions N: 60")
    print(f"Nombre de poids (N+1): 61")
    print(f"{'='*80}\n")
    
    # Paramètres d'apprentissage
    biais_range = [-1, 1]
    eta = 0.1
    maxIter = 10000
    
    # ========================================================================
    # PARTIE I: Apprentissage sur train, test sur test (et vice versa)
    # ========================================================================
    print(f"\n{'='*80}")
    print("PARTIE I: APPRENTISSAGE SUR TRAIN, TEST SUR TEST")
    print(f"{'='*80}\n")
    
    # 2. Apprentissage sur train, test sur test
    print("2. Apprentissage sur 'train', test sur 'test'")
    print("-" * 60)
    print("Note: Utilisation de l'algorithme perceptron ONLINE (justification basée sur TP1)")
    print("     - Plus efficace en mémoire pour de grands ensembles")
    print("     - Mise à jour immédiate des poids après chaque exemple")
    print("     - Convergence généralement plus rapide pour ce type de problème\n")
    
    # Initialisation et apprentissage
    w_init = f_init_rand(train_df_prepare, biais_range)
    w_trained, n_iter, training_errors = perceptron_online(w_init, train_df_prepare, eta, maxIter)
    
    # Calcul des erreurs
    Ea = error(train_df_prepare, w_trained)
    # Pour calculer Eg, on doit tester sur test mais sans utiliser les classes
    # On reconstruit test avec les classes pour le calcul d'erreur
    Eg = error(test_df_prepare, w_trained)
    
    print(f"Erreur d'apprentissage Ea: {Ea}/{len(train_df_prepare)} = {Ea/len(train_df_prepare)*100:.2f}%")
    print(f"Erreur de généralisation Eg: {Eg}/{len(test_df_prepare)} = {Eg/len(test_df_prepare)*100:.2f}%")
    print(f"Nombre d'itérations: {n_iter}")
    
    # Affichage des poids
    print_weights(w_trained, "Poids du perceptron (train -> test)")
    
    # Calcul des stabilités sur test
    stabilities_test = calculate_stability(test_df_prepare, w_trained)
    print(f"Stabilités calculées pour {len(stabilities_test)} exemples de test")
    print(f"Stabilité moyenne: {np.mean(stabilities_test):.6f}")
    print(f"Stabilité min: {np.min(stabilities_test):.6f}")
    print(f"Stabilité max: {np.max(stabilities_test):.6f}")
    
    # Graphique des stabilités
    plot_stabilities(stabilities_test, "Stabilités des exemples de test (apprentissage sur train)")
    
    # Visualisation 3D avec hyperplan séparateur
    print("\nVisualisation 3D avec hyperplan séparateur...")
    visualize_perceptron_3D(train_df, w_init, w_trained, 
                           "Hyperplan séparateur - Apprentissage sur train")
    plt.show()
    
    # 3. Apprentissage sur test, test sur train
    print("\n3. Apprentissage sur 'test', test sur 'train'")
    print("-" * 60)
    
    w_init2 = f_init_rand(test_df_prepare, biais_range)
    w_trained2, n_iter2, training_errors2 = perceptron_online(w_init2, test_df_prepare, eta, maxIter)
    
    Ea2 = error(test_df_prepare, w_trained2)
    Eg2 = error(train_df_prepare, w_trained2)
    
    print(f"Erreur d'apprentissage Ea: {Ea2}/{len(test_df_prepare)} = {Ea2/len(test_df_prepare)*100:.2f}%")
    print(f"Erreur de généralisation Eg: {Eg2}/{len(train_df_prepare)} = {Eg2/len(train_df_prepare)*100:.2f}%")
    print(f"Nombre d'itérations: {n_iter2}")
    
    print_weights(w_trained2, "Poids du perceptron (test -> train)")
    
    stabilities_train = calculate_stability(train_df_prepare, w_trained2)
    print(f"Stabilités calculées pour {len(stabilities_train)} exemples de train")
    print(f"Stabilité moyenne: {np.mean(stabilities_train):.6f}")
    print(f"Stabilité min: {np.min(stabilities_train):.6f}")
    print(f"Stabilité max: {np.max(stabilities_train):.6f}")
    
    plot_stabilities(stabilities_train, "Stabilités des exemples de train (apprentissage sur test)")
    
    # Visualisation 3D avec hyperplan séparateur
    print("\nVisualisation 3D avec hyperplan séparateur...")
    visualize_perceptron_3D(test_df, w_init2, w_trained2, 
                           "Hyperplan séparateur - Apprentissage sur test")
    plt.show()
    
    # ========================================================================
    # PARTIE II: Algorithme Pocket
    # ========================================================================
    print(f"\n{'='*80}")
    print("PARTIE II: ALGORITHME POCKET")
    print(f"{'='*80}\n")
    
    # Test avec différentes initialisations et deltas
    etas = [0.01, 0.1, 0.5, 1.0]
    initializations = ['random', 'hebb']
    target_error = 5  # Arrêter quand Ea <= 5
    
    results_pocket = []
    
    for init_type in initializations:
        for eta_pocket in etas:
            print(f"\n--- Initialisation: {init_type}, Delta (eta): {eta_pocket} ---")
            
            # Initialisation
            if init_type == 'random':
                w_init_pocket = f_init_rand(train_df_prepare, biais_range)
            else:  # hebb
                w_init_pocket = f_init_Hebb(train_df_prepare, biais_range)
            
            # Apprentissage avec Pocket
            w_pocket, Ea_pocket, n_iter_pocket, error_history = pocket_algorithm(
                train_df_prepare, w_init_pocket, eta_pocket, maxIter, target_error
            )
            
            # Test sur test
            Eg_pocket = error(test_df_prepare, w_pocket)
            
            results_pocket.append({
                'init': init_type,
                'eta': eta_pocket,
                'Ea': Ea_pocket,
                'Eg': Eg_pocket,
                'iterations': n_iter_pocket
            })
            
            print(f"Ea: {Ea_pocket}/{len(train_df_prepare)} = {Ea_pocket/len(train_df_prepare)*100:.2f}%")
            print(f"Eg: {Eg_pocket}/{len(test_df_prepare)} = {Eg_pocket/len(test_df_prepare)*100:.2f}%")
            print(f"Itérations: {n_iter_pocket}")
    
    # Test en échangeant train et test
    print(f"\n{'='*60}")
    print("Pocket: Apprentissage sur 'test', test sur 'train'")
    print(f"{'='*60}\n")
    
    for init_type in initializations:
        for eta_pocket in etas:
            print(f"\n--- Initialisation: {init_type}, Delta (eta): {eta_pocket} ---")
            
            if init_type == 'random':
                w_init_pocket = f_init_rand(test_df_prepare, biais_range)
            else:
                w_init_pocket = f_init_Hebb(test_df_prepare, biais_range)
            
            w_pocket, Ea_pocket, n_iter_pocket, error_history = pocket_algorithm(
                test_df_prepare, w_init_pocket, eta_pocket, maxIter, target_error
            )
            
            Eg_pocket = error(train_df_prepare, w_pocket)
            
            results_pocket.append({
                'init': init_type,
                'eta': eta_pocket,
                'Ea': Ea_pocket,
                'Eg': Eg_pocket,
                'iterations': n_iter_pocket,
                'train_set': 'test'  # marquer que c'est test->train
            })
            
            print(f"Ea: {Ea_pocket}/{len(test_df_prepare)} = {Ea_pocket/len(test_df_prepare)*100:.2f}%")
            print(f"Eg: {Eg_pocket}/{len(train_df_prepare)} = {Eg_pocket/len(train_df_prepare)*100:.2f}%")
            print(f"Itérations: {n_iter_pocket}")
    
    # Tableau récapitulatif
    print(f"\n{'='*80}")
    print("TABLEAU RÉCAPITULATIF - ALGORITHME POCKET")
    print(f"{'='*80}")
    print(f"{'Init':<10} {'Eta':<8} {'Ea':<8} {'Eg':<8} {'Itérations':<12}")
    print("-" * 80)
    for r in results_pocket:
        train_set = r.get('train_set', 'train')
        print(f"{r['init']:<10} {r['eta']:<8.2f} {r['Ea']:<8} {r['Eg']:<8} {r['iterations']:<12} ({train_set})")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # PARTIE III: Test si L (train+test) est linéairement séparable
    # ========================================================================
    print(f"\n{'='*80}")
    print("PARTIE III: TEST DE SÉPARABILITÉ LINÉAIRE DE L (train+test)")
    print(f"{'='*80}\n")
    
    w_init_L = f_init_rand(complete_df_prepare, biais_range)
    w_trained_L, n_iter_L, training_errors_L = perceptron_online(
        w_init_L, complete_df_prepare, eta, maxIter
    )
    
    Ea_L = error(complete_df_prepare, w_trained_L)
    
    print(f"Apprentissage sur L (train+test, {len(complete_df_prepare)} exemples)")
    print(f"Erreur d'apprentissage Ea: {Ea_L}/{len(complete_df_prepare)} = {Ea_L/len(complete_df_prepare)*100:.2f}%")
    print(f"Nombre d'itérations: {n_iter_L}")
    
    if Ea_L == 0:
        print("\n✓ L est LINÉAIREMENT SÉPARABLE (LS)")
        print("  Justification: Le perceptron a convergé avec Ea = 0, ce qui signifie")
        print("  que tous les exemples sont correctement classés par l'hyperplan séparateur.")
    else:
        print("\n✗ L n'est PAS linéairement séparable (non-LS)")
        print(f"  Justification: Le perceptron n'a pas convergé avec Ea = {Ea_L} > 0.")
        print("  Il existe au moins un exemple mal classé après convergence ou arrêt.")
    
    print_weights(w_trained_L, "Poids du perceptron sur L (train+test)")
    
    # ========================================================================
    # PARTIE IV: Early Stopping
    # ========================================================================
    print(f"\n{'='*80}")
    print("PARTIE IV: EARLY STOPPING")
    print(f"{'='*80}\n")
    
    # Répéter l'expérience plusieurs fois pour obtenir des statistiques
    n_experiments = 5
    results_early_stopping = []
    
    for exp in range(n_experiments):
        print(f"\n--- Expérience {exp+1}/{n_experiments} ---")
        
        # Mélanger les données
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
        w_init_ES = f_init_rand(LA, biais_range)
        
        # Early Stopping
        w_best_ES, best_Ev, n_iter_ES, train_errors_ES, val_errors_ES = early_stopping(
            LA, LV, w_init_ES, eta, maxIter, patience=20
        )
        
        # Calcul des erreurs
        Ea_ES = error(LA, w_best_ES)
        Ev_ES = best_Ev
        Et_ES = error(LT, w_best_ES)
        
        results_early_stopping.append({
            'Ea': Ea_ES,
            'Ev': Ev_ES,
            'Et': Et_ES,
            'iterations': n_iter_ES
        })
        
        print(f"Ea: {Ea_ES}/{len(LA)} = {Ea_ES/len(LA)*100:.2f}%")
        print(f"Ev: {Ev_ES}/{len(LV)} = {Ev_ES/len(LV)*100:.2f}%")
        print(f"Et: {Et_ES}/{len(LT)} = {Et_ES/len(LT)*100:.2f}%")
        print(f"Itérations: {n_iter_ES}")
    
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
    
    print("\n" + "="*80)
    print("FIN DU TP2")
    print("="*80)


