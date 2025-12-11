def X(n=2, d=2):
    """
    Génère toutes les combinaisons de `d` indices allant de 0 à n-1,
    sous forme de floats.
    """
    return [ [float(x) for x in t] for t in product(range(n), repeat=d) ]

def signe(x):
  if x > 0:
    return 1
  else:
    return -1


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

def f_init_Hebb(L_ens, biais):
  p = len(L_ens)
  dim = len(L_ens[0][0])

  w = []
  for i in range(dim):
    w.append(0.0)

  for k in range(p):
    w[0] = biais # Biais
    for i in range(1,dim):
      w[i] = w[i] + L_ens[k][0][i]*L_ens[k][1]

  return w

def f_init_rand(L_ens, biais):
  dim = len(L_ens[0][0])

  w = []
  for i in range(dim):
    w.append(0.0)

  w[0] = biais #Ajout du biais
  for i in range(1,dim):
    w[i] = (rd(-1,1))

  return w

def init_perceptron_prof(N, biais_range=(-10e6, 10e6)):
  w_prof = np.zeros(N+1)

  biais_min, biais_max = biais_range
  w_prof[0] = np.random.uniform(biais_min, biais_max)
  for i in range(1,len(w_prof)):
      w_prof[i] = (rd(biais_min, biais_max))
  return w_prof

def L_prof(X, w_prof):
  L_ens = []
  for x in X:
    L_ens.append([x,signe(np.dot(x, w_prof))])
  return L_ens

def init_many_perceptrons(qte, N, biais_range=(-10e6, 10e6)):
  # Initialisation : N poids par perceptron (SANS biais)
  #W_many_perceptrons = np.zeros((qte, N))
  borne_min, borne_max = biais_range
  W_many_perceptrons = np.random.uniform(borne_min, borne_max, size=(qte, N))

  # Créer un nouveau array avec N+1 colonnes (N poids + biais)
  W_avec_biais = np.zeros((qte, N + 1))

  biais_min, biais_max = biais_range

  for j in range(qte):
      # Mettre le biais en première position
      W_avec_biais[j, 0] = np.random.uniform(biais_min, biais_max)
      # Copier les anciens poids aux positions 1 à N+1
      W_avec_biais[j, 1:] = W_many_perceptrons[j]

  # Remplacer l'ancienne matrice
  return W_avec_biais

def create_points(nbPoints, N, bornes=(-10e6,10e6)): #Ajout de 1.0 au début pour le biais
  points = []
  borne_min, borne_max = bornes
  for p in range(nbPoints):
    point = []
    point.append(1.0) #Le biais
    for i in range(N):
      x_i = rd(borne_min, borne_max)
      point.append(x_i)
    points.append(point)

  return points

def recouvrement(vect1, vect2):
  produit_scalaire = np.dot(vect1, vect2)
  norme_vect_1 = np.linalg.norm(vect1)
  norme_vect_2 = np.linalg.norm(vect2)

  R = ( produit_scalaire / (norme_vect_1*norme_vect_2) )

  return R