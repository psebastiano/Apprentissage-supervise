def perceptron_online(w_vect, L_ens, eta):
  #print("in perceptron online")

  w_k = copy.deepcopy(w_vect)
  stop = 0
  while stop < 50:     # boucle sur les époques
    #print('\n'*2, '_'*20, '\n')
    #print("stop =", stop)
    nb_mal_classe = 0

    for k in range(len(L_ens)):   # boucle sur les exemples
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
  return w_k, stop-1


def perceptron_batch(w_vect, L_ens, eta):
  #print("in perceptron online")

  w_k = copy.deepcopy(w_vect)
  stop = 0

  delta_w = np.zeros(len(w_vect))

  while stop < 50:     # boucle sur les époques
    #print('\n'*2, '_'*20, '\n')
    #print("stop =", stop)
    nb_mal_classe = 0

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
  return w_k, stop-1

def train_perceptron (perceptron, L_ens, eta, algorithme):
  #modifier les deux algos pour qu'il retourne les nbIter
  if algorithme == "online":
    perceptron, nbIter = perceptron_online(perceptron, L_ens, eta)
  elif algorithme == "batch":
    perceptron, nbIter = perceptron_batch(perceptron, L_ens, eta)

  return perceptron, nbIter

def make_tirage(W_many_perceptrons, w_prof, eta, N, P, bornes, algorithme, points = None):

  local_W_many_perceptrons = copy.deepcopy(W_many_perceptrons)
  nb_perceptron = len(local_W_many_perceptrons)
  moyennes = np.zeros((nb_perceptron, 2))

  #Créer P points de dimension N+1 (pour le biais)
  if points is None:
    borne_min, borne_max = bornes
    points = create_points(P, N, bornes)

  #Création de l'ensemble d'apprentissage L_prof avec w_prof
  L_ens = L_prof(points, w_prof)

  #Entrainement en // des 20 Perceptrons : index 'l'
  threads = []
  results = [None] * nb_perceptron

  def worker(l):
    perceptron = local_W_many_perceptrons[l]
    new_weights, nbIter = train_perceptron(perceptron, L_ens, eta, algorithme)
    R = recouvrement(w_prof, new_weights)
    results[l] = (nbIter, R, new_weights)

  for l in range(nb_perceptron):
    t = th.Thread(target=worker, args=(l,))
    t.start()
    threads.append(t)

  for t in threads:
    t.join()

  for l in range(nb_perceptron):
    nbIter, R, new_w = results[l]
    local_W_many_perceptrons[l] = new_w
    #print("\n","-"*20, "\n")
    #print("nbIter :", nbIter)
    #print("R :", R)
    #print("\n","-"*20, "\n")
    moyennes[l, 0] = nbIter
    moyennes[l, 1] = R

  return local_W_many_perceptrons, moyennes, L_ens

def train_perceptron_worker(args):
    """
    Fonction enveloppe pour l'entraînement d'un seul perceptron dans un processus séparé.
    Retourne les résultats et le poids final.
    """
    # Dépaquetage des arguments
    initial_weights, L_ens, eta, algorithme, w_prof = args
    
    # Entraînement et calcul des métriques
    # On suppose que train_perceptron ne modifie pas initial_weights et retourne new_weights.
    new_weights, nbIter = train_perceptron(initial_weights, L_ens, eta, algorithme)
    R = recouvrement(w_prof, new_weights)
    
    # Retourne le résultat du travail : (nbIter, R, new_weights)
    return nbIter, R, new_weights

def make_tirage_multiprocessing(W_many_perceptrons, w_prof, eta, N, P, bornes, algorithme, points=None):
    """
    Exécute l'entraînement de plusieurs perceptrons en parallèle en utilisant des processus
    pour contourner le GIL.
    """
    
    # 1. PRÉPARATION ET PROTECTION DES DONNÉES
    
    # Créer une copie profonde immédiate de la liste des poids d'entrée. 
    # Cette copie locale est passée au Pool, l'original W_many_perceptrons n'est pas touché.
    local_W_many_perceptrons = copy.deepcopy(W_many_perceptrons)
    nb_perceptron = len(local_W_many_perceptrons)
    
    # Créer les points si non fournis.
    if points is None:
        points = create_points(P, N, bornes)
        
    # Création de l'ensemble d'apprentissage L_prof
    L_ens = L_prof(points, w_prof)

    # 2. PRÉPARATION DES ARGUMENTS POUR LE POOL
    
    # Création d'une liste d'arguments pour chaque perceptron
    # Le Pool se chargera de distribuer ces arguments aux processus worker.
    tasks = [
        (local_W_many_perceptrons[l], L_ens, eta, algorithme, w_prof)
        for l in range(nb_perceptron)
    ]

    # 3. EXÉCUTION EN PARALLÈLE (Multiprocessing Pool)
    
    # Utilisation d'un Pool de processus. Par défaut, il utilise le nombre de cœurs de CPU.
    # Le 'with' gère automatiquement la fermeture et la jointure des processus.
    with mp.Pool() as pool:
        # map() distribue les tâches et bloque jusqu'à ce que tous les résultats soient collectés
        # results est une liste d'éléments retournés par train_perceptron_worker.
        results = pool.map(train_perceptron_worker, tasks)

    # 4. AGRÉGATION DES RÉSULTATS
    
    moyennes = np.zeros((nb_perceptron, 2))
    new_weights_list = [None] * nb_perceptron
    
    for l, (nbIter, R, new_w) in enumerate(results):
        moyennes[l, 0] = nbIter
        moyennes[l, 1] = R
        new_weights_list[l] = new_w

    # Retourne les poids finaux (séparés de l'entrée), les métriques, et l'ensemble d'apprentissage
    return new_weights_list, moyennes, L_ens