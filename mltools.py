import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons, make_circles, make_classification
import numpy as np

def generate_classification_data(kind='moons', n_samples=300, noise=0.2, seed=42):
    np.random.seed(seed)
    if kind == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif kind == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    elif kind == 'linear':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1, random_state=seed)
    elif kind == 'chess':
        X=np.reshape(np.random.uniform(-4,4,2*n_samples),(n_samples,2))
        y=np.ceil(X[:,0])+np.ceil(X[:,1])
        y=2*(y % 2)-1
        y = np.where(y == -1, 0, 1)
    elif kind == 'xor':
        X=np.vstack((np.random.multivariate_normal([1,1],np.diag([noise,noise]),n_samples//4),np.random.multivariate_normal([-1,-1],np.diag([noise,noise]),n_samples//4)))
        xneg=np.vstack((np.random.multivariate_normal([-1,1],np.diag([noise,noise]),n_samples//4),np.random.multivariate_normal([1,-1],np.diag([noise,noise]),n_samples//4)))
        data=np.vstack((X,xneg))
        y=np.hstack((np.ones(n_samples//2),-np.ones(n_samples//2)))
    else:
        raise ValueError("Unknown kind of dataset")

    y = y.reshape(-1, 1)  # make it 2D for consistency
    return X, y

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
         #melange de 2 gaussiennes
         xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//2)
         xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//2)
         data=np.vstack((xpos,xneg))
         y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    y = np.where(y == -1, 0, 1)
    return data,y.reshape(-1, 1)

def plot_data(data,labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    if labels is not None:
        labels = labels.reshape(-1)
    cols,marks = ["red", "green", "blue", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        plt.scatter(data[:,0],data[:,1],marker="x")
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i])

def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def plot_frontiere(data,f,step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),colors=('gray','blue'),levels=[-1,0,1])


def plot_loss(loss_log):
    plt.plot(loss_log)
    plt.show()

def generate_linreg_data(n_samples=100, input_dim=2, noise=0.1, seed=42):
    np.random.seed(seed)
    X = np.random.rand(n_samples, input_dim)
    true_weights = np.random.randn(input_dim, 1)
    y = X @ true_weights + noise * np.random.randn(n_samples, 1)
    return X, y, true_weights


def plot_2d_predictions(X, y, model):
    if X.shape[1] != 1:
        raise ValueError("plot_2d_predictions only works for input_dim = 1")

    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.forward(X_plot)

    plt.scatter(X, y, label="True Data")
    plt.plot(X_plot, y_pred, color='red', label="Model Prediction")
    plt.legend()
    plt.title("Linear Regression (1D input)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def plot_3d_predictions(X, y, model):
    if X.shape[1] != 2:
        raise ValueError("plot_3d_predictions only works for input_dim = 2")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Affichage des points de données
    ax.scatter(X[:, 0], X[:, 1], y.flatten(), color='blue', label='True Data')

    # Grille pour afficher le plan prédit
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    y_pred = model.forward(X_grid).reshape(x1_grid.shape)

    # Affichage du plan
    ax.plot_surface(x1_grid, x2_grid, y_pred, color='red', alpha=0.5, label='Model Prediction')

    ax.set_title("Linear Regression (2D input)")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    plt.show()

def plot_decision_boundary(X, y, model, activation=None, title="Decision Boundary"):
    sns.set_theme(style="whitegrid")
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    if activation is not None:
        zz = activation.forward(model.forward(grid)).reshape(xx.shape)
    else:
        zz = model.forward(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, cmap="RdBu_r", alpha=0.6, levels=10)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y.flatten(), palette="coolwarm", edgecolor="k")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()