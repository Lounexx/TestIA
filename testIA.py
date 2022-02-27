import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


# Initialisation du modèle
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))



def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A


# Calcul le taux d'erreur comparé au résultat attendu
def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))



def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)



def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)


def prediction(X, W, b):
    A = model(X, W, b)
    return A >= 0.5

    

def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):
    W, b = initialisation(X)

    Loss = []
    # Entraînement des neurones sur le modèle pour un nombre d'itération
    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = prediction(X, W, b)

    # Afficher courbe d'erreur

    #plt.plot(Loss)
    #plt.show()

    return (W, b)
    

W,b = artificial_neuron(X,y)

plante_a_tester = np.array([1,6])

# Création du graphique
plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
plt.scatter(plante_a_tester[0], plante_a_tester[1], c='r')

if(bool(prediction(plante_a_tester,W,b))):
    print("La plante est toxique")
else :
    print("La plante n'est pas toxique")


plt.show()



