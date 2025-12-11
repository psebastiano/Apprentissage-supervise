from random import uniform as rd
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import copy
import threading as th
import multiprocessing as mp
import pandas as pd
import time
from IPython.display import display, HTML
import dataframe_image as dfi

from functions import *
from perceptrons import *
from training import *
from plotting_functions import *

N = 2
X_ens = X(2, N)
X_ens_N_1 = [[1.0] + x for x in X_ens] #Ajout de x_0 pour le biais à tous les points

L_OU = L(f_OU, X_ens_N_1)
L_AND = L(f_AND, X_ens_N_1)
L_XOR = L(f_XOR, X_ens_N_1)

print(L_OU)
print(L_AND)
print(L_XOR)



biais = 1.0
eta = 0.14
L_ens = L_XOR

w_k = f_init_rand(L_OU, biais)
plot_L_et_w_bias_first(L_ens, w_k)


w_k,_ = perceptron_online(w_k, L_ens, eta)

print("Poids finaux (biais en premier):", w_k)
plot_L_et_w_bias_first(L_ens, w_k)


w_k,_ = perceptron_batch(w_k, L_ens, eta)

print("Poids finaux (biais en premier):", w_k)
plot_L_et_w_bias_first(L_ens, w_k)



#Perceptron Professeur
N = 2
X_prof = X(2, N)

X_prof_N_1 = [[1.0] + x for x in X_prof] #Ajout de x_0 pour le biais à tous les points


w_prof = init_perceptron_prof(N)
L_ens = L_prof(X_prof_N_1, w_prof)

plot_L_et_w_bias_first(L_ens, w_prof)

qte = 20
biais_range = (-20, 20)
W_perceptron_batch = init_many_perceptrons(qte, N, biais_range)


P = 100
N = 2

bornes = (-10,10)

#Créer P points de dimension N+1 (pour le biais)
points = create_points(P, N, bornes)

w_prof = init_perceptron_prof(N, bornes)
L_ens = L_prof(points, w_prof)




perceptron = W_perceptrons[1]

perceptron, nbIter = train_perceptron(perceptron, L_ens, eta, "online")
print(perceptron)
print(nbIter)

print("Poids finaux (biais en premier):", w_k)
plot_L_et_w_bias_first(L_ens, perceptron)




# --- Important : Lors de l'appel, assurez-vous que le code est dans un bloc if __name__ == "__main__": ---
# Cela est crucial pour multiprocessing sur Windows et dans les notebooks.
# Exemple d'appel :
# if __name__ == "__main__":
#     # Votre code d'exécution ici
#     W_final, moy, L_set = make_tirage_multiprocessing(...)

# %%
N = 2
P = 10
nb_perceptrons = 20

biais_range = (-20,20)

W_perceptrons = init_many_perceptrons(nb_perceptrons, N, biais_range)
w_prof = init_perceptron_prof(N, biais_range)

eta0 = 0.8

# %%
points = create_points(P, N, biais_range)

# %%
algorithme = "online"
trained_perceptrons, moyennes, L_ens = make_tirage(W_perceptrons, w_prof, eta, N, P, biais_range, algorithme, points)

# %%
all_array = np.vstack([w_prof,trained_perceptrons])

x_lim = 20
y_lim = 20
plot_L_et_w_multiple_fixed_axes(L_ens, all_array, (-x_lim, x_lim), (-y_lim, y_lim))


# %%
df_nbIter=pd.DataFrame(moyennes,columns=['Nbre_iter_convergence', 'Recouvrement R'])
df_nbIter.index.name = 'Perceptron #'
print(df_nbIter)

Nb_iter_moyen, R_moyen = np.mean(moyennes[:, 0]), np.mean(moyennes[:, 1])
print("-"*20, '\n' )
print("Nb_iter_moyen :", Nb_iter_moyen)
print("R_moyen :", R_moyen)

# %%
algorithme = "batch"
trained_perceptrons, moyennes, L_ens = make_tirage(W_perceptrons, w_prof, eta, N, P, biais_range, algorithme, points)

# %%
all_array = np.vstack([w_prof,trained_perceptrons])

x_lim = 20
y_lim = 20
plot_L_et_w_multiple_fixed_axes(L_ens, all_array, (-x_lim, x_lim), (-y_lim, y_lim))

# %%
df_nbIter=pd.DataFrame(moyennes,columns=['Nbre_iter_convergence', 'Recouvrement R'])
df_nbIter.index.name = 'Perceptron #'
print(df_nbIter)

Nb_iter_moyen, R_moyen = np.mean(moyennes[:, 0]), np.mean(moyennes[:, 1])
print("-"*20, '\n' )
print("Nb_iter_moyen :", Nb_iter_moyen)
print("R_moyen :", R_moyen)

# %%
#Meilleure convergence avec la méthode batch

# %%
eta1 = eta0/2
eta2 = eta0/10

algos = ['online','batch']
tests_conf_eta = [eta0, eta1, eta2]
tests_conf_P = [10, 100, 500, 1000]
tests_conf_N = [2, 10, 100, 5000]

param_combinaison = list(product(tests_conf_eta, algos, tests_conf_P, tests_conf_N))

print(param_combinaison)

column_names = ['eta', 'Algorithme', 'Nombre de points P', 'Dimension N']

#Creation du dataframe pour le stockage des paramètres de test
results = pd.DataFrame(param_combinaison, columns=column_names)

#Ajout des deux colonnes pour le stockage des moyennes
results['Avg_nbIter'] = np.nan
results['Avg_R'] = np.nan

print(f"Dimensions du DataFrame (lignes, colonnes) : {results.shape}")
print("-" * 50)
print("Aperçu du DataFrame de Configuration :")
print(results)

# %%
def make_n_tirages(eta, P, N, nb_tirage, nb_perceptrons, init_range, algo1, algo2):
    #20 Perceptrons aléatoire per paire (N/P) : ils sont initialisés avant la boucle des 100 tirages
    W_perceptrons = init_many_perceptrons(nb_perceptrons, N, init_range)
    w_prof = init_perceptron_prof(N, init_range)

    nb_iter_online_temp = []
    R_online_temp = []
    temps_online_total = 0.0

    nb_iter_batch_temp = []
    R_batch_temp = []
    temps_batch_total = 0.0

    for tir in range(nb_tirage):
        #print("tirage n : ", tir, '-'*10, "checkpoint 1")
        #Creation d'un point set per tirage
        points = create_points(P, N, init_range)
        #print("tirage n : ", tir, '-'*10, "checkpoint 2")
        
        #Appel fonction d'entraînement des 20 perceptrons en parallèle avec les deux algortime
        temps_start_online = time.perf_counter()
        algo = 'online'
        _, moyennes_online, _ = make_tirage(W_perceptrons, w_prof, eta, N, P, init_range, algo1, points)
        temps_online_total += (time.perf_counter() - temps_start_online)
        #print("tirage n : ", tir, '-'*10, "checkpoint 3")
        
        nb_iter_online_temp.append(np.mean(moyennes_online[:, 0])) #Moyenne sur les 20 perceptrons entraîné sur le tirage
        R_online_temp.append(np.mean(moyennes_online[:, 1]))
        #print("tirage n : ", tir, '-'*10, "checkpoint 4")
        
        temps_start_batch = time.perf_counter()
        algo = 'batch'
        _, moyennes_batch, _ = make_tirage(W_perceptrons, w_prof, eta, N, P, init_range, algo2, points)
        nb_iter_batch_temp.append(np.mean(moyennes_batch[:, 0])) #Idem
        R_batch_temp.append(np.mean(moyennes_batch[:, 1]))
        temps_batch_total += (time.perf_counter() - temps_start_batch)
        #print("tirage n : ", tir, '-'*10, "checkpoint 5")
        
    #Moyenne sur les 100 tirages
    nb_iter_online_moyen = np.mean(nb_iter_online_temp)
    R_online_moyen = np.mean(R_online_temp)

    nb_iter_batch_moyen = np.mean(nb_iter_batch_temp)
    R_iter_batch_moyen = np.mean(R_batch_temp)

    return (nb_iter_online_moyen, R_online_moyen, temps_online_total,
            nb_iter_batch_moyen, R_iter_batch_moyen, temps_batch_total)

# %%
eta = eta0
P = 10
N = 2

NB_TIRAGE = 100
NB_PERCEPTRONS = 20
BIAIS_RANGE = (-20, 20)

(nb_iter_online, R_online, temps_online,
     nb_iter_batch, R_batch, temps_batch) = make_n_tirages(
        eta=eta,
        P=P,
        N=N,
        nb_tirage=NB_TIRAGE,
        nb_perceptrons=NB_PERCEPTRONS,
        init_range=BIAIS_RANGE, # Correspond à 'biais_range' dans la fonction
        algo1='online',
        algo2='batch'
    )

print("nb_iter_online : ", nb_iter_online,"\n",
    " R_online : ", R_online,"\n",
    "temps_total_online:", temps_online, "\n",
    " nb_iter_batch : ", nb_iter_batch,"\n",
    " R_batch: ", R_batch,"\n",
    "temps_total_batch:", temps_batch
)

# %%


# %%
tests_conf_eta = [eta0, eta0/2, eta0/10]
tests_conf_P = [10, 100, 500, 1000]
tests_conf_N = [2, 10, 100, 5000]

# Paramètres globaux pour la simulation
NB_TIRAGE = 30
NB_PERCEPTRONS = 20
BIAIS_RANGE = (-20, 20)

# Création d'un DataFrame temporaire contenant uniquement les combinaisons uniques (eta, P, N)
# Nous filtrons les algorithmes car make_n_tirages les gère en interne.
unique_combinations = pd.DataFrame(
    list(product(tests_conf_eta, tests_conf_P, tests_conf_N)),
    columns=['eta', 'Nombre de points P', 'Dimension N']
)

# Initialisation du DataFrame de résultats final
# Le DataFrame final doit contenir des lignes distinctes pour 'online' et 'batch'.
results_data = []

# --- 2. Boucle de Simulation ---

print("Démarrage des simulations (un appel à make_n_tirages pour chaque combinaison eta/P/N)...")

for index, row in unique_combinations.iterrows():
    eta = row['eta']
    P = int(row['Nombre de points P'])
    N = int(row['Dimension N'])

    print(f"\n--- Running: eta={eta:.4f}, P={P}, N={N} ---")

    # Appel de la fonction pour obtenir les moyennes des 100 tirages pour les deux algos
    (nb_iter_online, R_online, temps_online,
     nb_iter_batch, R_batch, temps_batch) = make_n_tirages(
        eta=eta,
        P=P,
        N=N,
        nb_tirage=NB_TIRAGE,
        nb_perceptrons=NB_PERCEPTRONS,
        init_range=BIAIS_RANGE, # Correspond à 'biais_range' dans la fonction
        algo1='online',
        algo2='batch'
    )

    # 3. Stocker les résultats dans un format adapté au DataFrame final
    # On ajoute DEUX lignes pour chaque appel : une pour 'online' et une pour 'batch'
    
    # Résultat Online
    results_data.append({
        'eta': eta,
        'Algorithme': 'online',
        'Nombre de points P': P,
        'Dimension N': N,
        'Avg_nbIter': nb_iter_online,
        'Avg_R': R_online,
        'Temps_execution_s': temps_online,
    })

    # Résultat Batch
    results_data.append({
        'eta': eta,
        'Algorithme': 'batch',
        'Nombre de points P': P,
        'Dimension N': N,
        'Avg_nbIter': nb_iter_batch,
        'Avg_R': R_batch,
        'Temps_execution_s': temps_batch
    })






# %%
print(results_data)

# %%
# --- 4. Création du DataFrame final - Avant post traitement---

# Convertir la liste de résultats en DataFrame
df_results = pd.DataFrame(results_data)

print("\nSimulations terminées.")
print(f"DataFrame de résultats final créé. Shape : {df_results.shape}")
print(df_results.head())

# Sauvegarder les résultats dans un fichier CSV
df_results.to_csv('perceptron_results_saved_30_iter.csv', index=False)
print("DataFrame sauvegardé en 'perceptron_results_saved_30_iter.csv'.")

# %%
print(NB_TIRAGE)
#Post traitement pour récuperer le temps par tirage (1 tirage = 20 perceptrons entraîné en parallèle)
result_post_traitement = results_data
for result in result_post_traitement:
    result['Temps_execution_s'] =  result['Temps_execution_s'] / NB_TIRAGE


# --- 4. Création du DataFrame final après post traitement ---

# Convertir la liste de résultats en DataFrame
df_results_post_traitement = pd.DataFrame(result_post_traitement)

print("\nSimulations terminées.")
print(f"DataFrame de résultats final créé. Shape : {df_results_post_traitement.shape}")
print(df_results.head())

# Sauvegarder les résultats dans un fichier CSV
df_results_post_traitement.to_csv('perceptron_results_saved_30_iter_apres_post_traitement.csv', index=False)
print("DataFrame sauvegardé en 'perceptron_results_saved_30_iter.csv'.")


# %%
df_results_imported = pd.read_csv('perceptron_results_saved_30_iter.csv')
print("DataFrame 'perceptron_results_saved_30_iter_apres_post_traitement.csv' rechargé avec succès depuis le fichier CSV.")
print(f"Dimensions : {df_results_imported.shape}")

# %%
# --- FONCTION D'AFFICHAGE DE LÉGENDE (COLORBAR) ---
def display_colorbar(cmap_name, vmin, vmax, label, filename=None, figsize=(10, 0.4)):
    """
    Affiche une Colorbar simple pour servir de légende à la carte thermique et la sauvegarde en JPG.
    """
    cmap = plt.colormaps[cmap_name]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal',
        label=label
    )
    plt.tight_layout()
    
    # --- AJOUT DE LA LOGIQUE DE SAUVEGARDE ---
    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"[Sauvegardé] : {filename}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de {filename}: {e}")
            
    plt.show() # Affiche le graphique dans le notebook après la sauvegarde (si non sauv. il était déjà affiché)

def create_styled_comparison_table(df_results, current_eta, all_etas_min_max):
    """
    Crée trois tableaux de contingence N x P (R, NbIter, Temps) consolidés
    (Online vs Batch) pour un ETA donné, et applique une coloration cohérente
    basée sur les min/max globaux.
    """
    
    # Filtrer par ETA courant
    df_eta = df_results[df_results['eta'] == current_eta].copy()
    
    # Définition des métriques et des labels (DICTIONNAIRE)
    metrics = {
        'Avg_R': 'Recouvrement R',
        'Avg_nbIter': 'Nb. Itérations Moy.',
        'Temps_execution_s': 'Temps (s)'
    }
    
    styled_results = {}

    for metric_col, metric_label in metrics.items():
        
        # --- Création du tableau pivot N x P consolidé (Online + Batch) ---
        pivot_online = df_eta[df_eta['Algorithme'] == 'online'].pivot_table(
            index='Dimension N', columns='Nombre de points P', values=metric_col
        )
        pivot_batch = df_eta[df_eta['Algorithme'] == 'batch'].pivot_table(
            index='Dimension N', columns='Nombre de points P', values=metric_col
        )
        
        # Concaténation des deux tableaux côte à côte
        comparison_df = pd.concat(
            [pivot_online, pivot_batch], 
            axis=1, 
            keys=['Online', 'Batch'] # Macro-colonnes
        )
        
        styled_df = comparison_df.style
        
        # --- Récupérer les min/max globaux pour une échelle cohérente ---
        vmin = all_etas_min_max[metric_col]['min']
        vmax = all_etas_min_max[metric_col]['max']
        
        # --- Application des Couleurs et du Formatage ---
        
        if metric_col == 'Avg_R':
            # R: Vert (bien) à Rouge (mal). Inversion de 'RdYlGn'.
            cmap_name = 'RdYlGn' 
            styled_df = styled_df.background_gradient(
                cmap=cmap_name, vmin=vmin, vmax=vmax
            ).format(formatter='{:.3f}')
            
        elif metric_col == 'Avg_nbIter':
            # NbIter: Clair (faible) à Sombre (élevé).
            cmap_name = 'YlOrBr' 
            styled_df = styled_df.background_gradient(
                cmap=cmap_name, vmin=vmin, vmax=vmax
            ).format(formatter='{:.0f}')
            
        elif metric_col == 'Temps_execution_s':
            # Temps: Clair (rapide) à Sombre (lent).
            cmap_name = 'viridis' 
            styled_df = styled_df.background_gradient(
                cmap=cmap_name, vmin=vmin, vmax=vmax
            ).format(formatter='{:.4f}')

        # --- Définir la largeur des colonnes via CSS pour uniformité ---
        styles = [
            # Applique la largeur à toutes les colonnes sauf l'index
            {'selector': 'th:not(.index_name)', 'props': [('width', '80px')]}
        ]
        styled_df = styled_df.set_table_styles(styles)

        # Mise à jour des noms de colonnes pour l'affichage
        styled_df.columns.names = ['Algorithme', 'P']
        styled_df = styled_df.set_caption(f"Comparaison N x P pour {metric_label} (ETA={current_eta:.4f})")
        
        # Stockage du Styler et des informations de légende
        styled_results[metric_col] = (styled_df, cmap_name, vmin, vmax, metric_label)
        
    return styled_results

# %%
df_results = df_results_imported

# ----------------------------------------------------------------------
# 1. PRÉPARATION : CALCUL DES MIN/MAX GLOBAUX (NON MODIFIÉ)
# ----------------------------------------------------------------------
# (Assurez-vous que cette section est exécutée)
metrics_list = ['Avg_R', 'Avg_nbIter', 'Temps_execution_s']
all_etas_min_max = {}
for metric in metrics_list:
    all_etas_min_max[metric] = {
        'min': df_results[metric].min(),
        'max': df_results[metric].max()
    }
# ----------------------------------------------------------------------

# Liste des taux d'apprentissage uniques (vos 0.8, 0.4, 0.08...)
all_etas = df_results['eta'].unique()
metrics_order = ['Avg_R', 'Avg_nbIter', 'Temps_execution_s']

for eta in all_etas:
    
    # Affichage du titre principal pour ce bloc d'ETA
    display(HTML(f"<h2>--- ANALYSE CONSOLIDÉE POUR ETA = {eta:.4f} ---</h2>"))
    
    styled_tables_info = create_styled_comparison_table(df_results, eta, all_etas_min_max)
    
    for metric in metrics_order:
        
        styled_df, cmap_name, vmin, vmax, metric_label = styled_tables_info[metric]
        
        # --- CRÉATION DU NOM DE FICHIER UNIQUE ---
        eta_str = f"{eta:.4f}".replace('.', '_')
        base_filename = f"eta_{eta_str}_{metric}"
        
        table_filename = f"{base_filename}_table.png"
        legend_filename = f"{base_filename}_legend.png"
        
        # 1. EXPORTATION DU TABLEAU STYLISÉ EN PNG (nécessite dataframe-image)
        try:
            dfi.export(styled_df, table_filename, table_conversion="matplotlib", dpi=300)
            print(f"[Sauvegardé Tableau] : {table_filename}")
        except Exception as e:
            print(f"ATTENTION: Échec de l'exportation du tableau en image. Assurez-vous d'avoir installé 'dataframe-image'. Erreur: {e}")

        # 1b. Afficher le tableau dans le notebook (en HTML)
        display(styled_df)
            
        # 2. EXPORTATION DE LA LÉGENDE (Colorbar) EN PNG
        display_colorbar(
            cmap_name, 
            vmin, 
            vmax, 
            f"Légende pour {metric_label} (Min={vmin:.3f} | Max={vmax:.3f})",
            filename=legend_filename, # <--- Sauvegarde du fichier de légende
            figsize=(10, 0.8)
        )
        
    print("\n" + "=" * 80 + "\n")

# %%



