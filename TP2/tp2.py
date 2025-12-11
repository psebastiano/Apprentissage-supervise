import pandas as pd
import numpy as np
from random import uniform as rd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    p = len(L_ens)
    dim = len(L_ens[0][0])

    borne_inf = biais_range[0]
    borne_sup = biais_range[1]

    w = []
    for i in range(dim):
        w.append(0.0)

    for k in range(p):
        w[0] = rd(borne_inf,borne_sup) # Biais
        for i in range(1,dim):
            w[i] = w[i] + L_ens[k][0][i]*L_ens[k][1]

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
    print('perceptron avant entraintement : ', w_k)
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


def project_data_on_2D(data):
    X = np.array(data['Donnee'].tolist())
    y = data['Classe voulue'].map({'M': 1, 'R': -1}).values # Utilisons les étiquettes numériques

    X_features = X[:, 1:]
    X_bias_col = X[:, :1]

    scaler = StandardScaler()
    X_scaled_features = scaler.fit_transform(X_features)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled_features)

    X_mines = X_2d[y == 1]
    X_rocks = X_2d[y == -1]

    plt.figure(figsize=(10, 8))

    # Plotter les mines
    plt.scatter(X_mines[:, 0], X_mines[:, 1], 
                label='Mine (M)', 
                marker='o', 
                color='red', 
                alpha=0.6)

    # Plotter les roches
    plt.scatter(X_rocks[:, 0], X_rocks[:, 1], 
                label='Roche (R)', 
                marker='x', 
                color='blue', 
                alpha=0.6)

    plt.xlabel(f'Composante Principale 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Composante Principale 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.title('Projection 2D du Sonar Data (Mines vs Roches) via PCA')
    plt.legend()
    plt.grid(True)
    plt.show()

def draw_data_points_3D(ax, data_df, pca_model=None, scaler_model=None):
    """
    Projette les données dans l'espace 3D PCA et trace les points sur l'axe donné.
    Gère la standardisation et l'ajustement de la PCA si les modèles ne sont pas fournis.
    
    Returns: pca_model, scaler_model
    """
    
    # Préparation des données
    X = np.array(data_df['Donnee'].tolist())
    y = data_df['Classe voulue'].map({'M': 1, 'R': -1}).values
    X_features = X[:, 1:]
    
    # 1. Standardisation et PCA
    if scaler_model is None:
        scaler = StandardScaler()
        X_scaled_features = scaler.fit_transform(X_features)
    else:
        scaler = scaler_model
        X_scaled_features = scaler.transform(X_features)
        
    if pca_model is None:
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X_scaled_features)
    else:
        pca = pca_model
        X_3d = pca.transform(X_scaled_features)

    # Séparer les points selon les classes
    X_mines = X_3d[y == 1]
    X_rocks = X_3d[y == -1]

    # Tracé des points
    ax.scatter(X_mines[:, 0], X_mines[:, 1], X_mines[:, 2],
               label='Mine (M)', marker='o', color='red', alpha=0.6)
    ax.scatter(X_rocks[:, 0], X_rocks[:, 1], X_rocks[:, 2],
               label='Roche (R)', marker='x', color='blue', alpha=0.6)
               
    # Mise à jour des labels des axes (pour un premier appel)
    variance_ratios = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC 1 ({variance_ratios[0]*100:.2f}%)')
    ax.set_ylabel(f'PC 2 ({variance_ratios[1]*100:.2f}%)')
    ax.set_zlabel(f'PC 3 ({variance_ratios[2]*100:.2f}%)')
    
    return pca, scaler

def draw_perceptron_plane_3D(ax, data_df, W_trained, pca_model, scaler_model, color='green', label=None):
    """
    Trace le plan de séparation Perceptron sur l'axe donné.
    
    Args:
        ax (Axes3D): L'objet Axes3D existant.
        data_df (pd.DataFrame): DataFrame des données (pour déterminer les limites du plan).
        W_trained (np.ndarray): Vecteur de poids entraîné (61 dimensions).
        pca_model (PCA): Le modèle PCA ajusté sur les données.
        scaler_model (StandardScaler): Le modèle StandardScaler ajusté.
    """
    W_trained=np.asarray(W_trained, dtype=np.float64)
    X = np.array(data_df['Donnee'].tolist())
    X_features = X[:, 1:] 
    
    # Transforme les features pour déterminer les limites du plan
    X_scaled_features = scaler_model.transform(X_features)
    X_3d = pca_model.transform(X_scaled_features)

    # Projection du vecteur de poids W dans l'espace PCA
    W_pca = pca_model.transform(W_trained[1:].reshape(1, -1))[0]
    W_pca_full = np.array([W_trained[0], W_pca[0], W_pca[1], W_pca[2]])
    
    # Définition de la grille pour le plan
    x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
    y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5
    
    PC1_grid, PC2_grid = np.meshgrid(np.linspace(x_min, x_max, 50),
                                     np.linspace(y_min, y_max, 50))
                                     
    # Calcul de PC3 (Z) : PC3 = (-w0 - w1*PC1 - w2*PC2) / w3
    if abs(W_pca_full[3]) < 1e-10:
        print("[AVERTISSEMENT] Le poids PC3 est trop petit pour tracer le plan.")
        return

    PC3_grid = (-W_pca_full[0] - W_pca_full[1] * PC1_grid - W_pca_full[2] * PC2_grid) / W_pca_full[3]

    # Tracé du Plan de Séparation
    ax.plot_surface(PC1_grid, PC2_grid, PC3_grid, 
                    alpha=0.2, color=color, label=label) # Retiré le label pour le mettre dans la légende custom
    

def project_data_on_3D(data):
    # 1. Préparation et standardisation des données (Identique au 2D)
    X = np.array(data['Donnee'].tolist())
    y = data['Classe voulue'].map({'M': 1, 'R': -1}).values

    # On exclut la colonne de biais (1.0) pour la standardisation et la PCA
    X_features = X[:, 1:]
    
    scaler = StandardScaler()
    X_scaled_features = scaler.fit_transform(X_features)

    # 2. Application de la PCA : n_components=3
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled_features) # X_3d a maintenant 3 colonnes

    # Séparer les points selon les classes
    X_mines = X_3d[y == 1]
    X_rocks = X_3d[y == -1]
    
    # 3. Setup de la figure et des axes 3D
    fig = plt.figure(figsize=(12, 10))
    # Création des axes avec la projection 3D
    ax = fig.add_subplot(111, projection='3d') 

    # Plotter les mines
    # Ajout de la troisième coordonnée : X_mines[:, 2]
    ax.scatter(X_mines[:, 0], X_mines[:, 1], X_mines[:, 2],
               label='Mine (M)',
               marker='o',
               color='red',
               alpha=0.6)

    # Plotter les roches
    # Ajout de la troisième coordonnée : X_rocks[:, 2]
    ax.scatter(X_rocks[:, 0], X_rocks[:, 1], X_rocks[:, 2],
               label='Roche (R)',
               marker='x',
               color='blue',
               alpha=0.6)

    # 4. Mise à jour des labels (Ajout de l'axe Z)
    variance_ratios = pca.explained_variance_ratio_
    
    ax.set_xlabel(f'Composante Principale 1 ({variance_ratios[0]*100:.2f}%)')
    ax.set_ylabel(f'Composante Principale 2 ({variance_ratios[1]*100:.2f}%)')
    # Label pour l'axe Z
    ax.set_zlabel(f'Composante Principale 3 ({variance_ratios[2]*100:.2f}%)')
    
    ax.set_title('Projection 3D du Sonar Data (Mines vs Roches) via PCA')
    ax.legend()
    
    # Utilisez savefig() si vous souhaitez sauvegarder l'image
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



if __name__=="__main__":

    #IMPORTATION DONNEES

    raw_dict = parse_custom_file('./sonar.mines', 9)
    mines_dict = clean_sonar_dict(raw_dict)


    # # l0=60
    # # c = 0
    # for key in sonar_dict.keys():
    #     print(key)
        
    #     l1=len(sonar_dict[f'{key}'])
    #     if l1 != l0:
    #         print(f'error. line {l1} différente de la précédente')
    #         print(sonar_dict[f'{key}'])
    #     c+=1
    # print(c)

    raw_dict = parse_custom_file('./sonar.rocks', 8) #Ce fichier n'a que 8 lignes au début
    rocks_dict = clean_sonar_dict(raw_dict)

    # c=0
    # for key in sonar_dict.keys():
    #     print(key)
        
    #     l1=len(sonar_dict[f'{key}'])
    #     if l1 != l0:
    #         print(f'error. line {l1} différente de la précédente')
    #         print(sonar_dict[f'{key}'])
    #     c+=1
    # print(c)


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

    #Ajout du baisi aux entrée (toujours = 1)
    for index, row in data_df.iterrows():
        data = row['Donnee'] 
        data_with_biais = [1.0] + data
        row['Donnee'] = data_with_biais

    #Séparation des données en ensemble de test et de train
    train_list = []
    test_list = []

    for index, row in data_df.iterrows():
        if '*' in row['ID']:
            train_list.append(row)
        else:
            test_list.append(row)

    test_df = pd.DataFrame(train_list)
    train_df = pd.DataFrame(test_list)

    #PRETRAITEMENT

    # 1 - Remove Col ID
    test_df_prepare = test_df.drop('ID', axis=1)
    train_df_prepare = train_df.drop('ID', axis=1)

    # 2 - Map classe voulue to -1 / +1
    mapping_classes = {'M': 1, 'R': -1}
    train_df_prepare['Classe voulue'] = train_df_prepare['Classe voulue'].map(mapping_classes)
    test_df_prepare['Classe voulue'] = test_df_prepare['Classe voulue'].map(mapping_classes)

    #DEBUT ENTRAINTEMENT
    #Initialisation d'un perceptron
    biais_range = [-1, 1]
    perceptron = f_init_rand(train_df_prepare, biais_range)
    
    #Entrainement
    eta = 0.1
    maxIter = 2000
    trained_perceptron, _ ,training_errors = perceptron_online(perceptron, train_df_prepare, eta, maxIter) #La fonction retourne aussi le nombre d'itérations
    n()
    print('traininge error : ', training_errors) 
    #Affichage porjection 3D pour visualisation de la convergence
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    pca, scaler = draw_data_points_3D(ax, train_df)
    draw_perceptron_plane_3D(ax, data_df, perceptron, pca, scaler, color='red', label='Perceptron état initial')
    draw_perceptron_plane_3D(ax, data_df, trained_perceptron, pca, scaler, color='blue', label='Perceptron état final')
    ax.set_title('Etat initial et Séparation du Perceptron dans l\'Espace PCA 3D')
    ax.legend()

    #Set to True to plot
    show(True)

    #Calcul de l'erreur
    training_error = error(train_df_prepare, trained_perceptron)
    print('training_error : ', training_error)