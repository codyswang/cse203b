from typing import Union, List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


#######################################
##      dataset helper functions     ##
#######################################

# NOTE: need to run 'pip install ucimlrepo' 
# for dataset retrieval using these functions


name_to_id = {
    "ionosphere": 52,
    "sonar": 151,
    # "bupa liver": 60, -- regression type dataset, omitted
    "balance scale": 12,
    "wine": 109,
    "iris": 53,
    "seeds": 236,
    "knowledge": 257, # updated since paper publication
    # 'seeds' doesn't come with ucimlrepo, 
    # so needs to be read from included txt file
}

# 5 is all we need for these datasets
COLOR_PALETTE = ["red", "blue", "green", "pink", "cyan"]

# dataset fetchers # 
def get_UCI_dataset(name: str, essentials_only: bool=True) -> Union[Dict, pd.DataFrame]:
    """
    Fetches a UCI dataset given the name as described
    in the paper.\\
    NOTE: Available datasets:
    - "ionosphere"
    - "balance scale"
    - "wine"
    - "iris"
    - "sonar"
    - "seeds"
    - "knowledge"

    Args:
        name (str): name of dataset
        essentials_only (bool): if True, only retrieves features and targets

    Returns:
        pd.DataFrame: selected dataset (if exists)
    """
    if name.lower() == "seeds": # only dataset not included
        data = np.loadtxt("seeds_dataset.txt")
        features = data[:, :-1].astype(np.float32)
        labels = data[:, -1].astype(int) - 1 # minus 1 to keep 0-indexed classes
        return features, labels

    dataset = fetch_ucirepo(id=name_to_id[name.lower()])
    if essentials_only:
        dataset = (dataset.data['features'].to_numpy(),
                   enum_string_array(dataset.data['targets']))
    return dataset


def get_synthetic_dataset(
    dim: int,
    seed: int=42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synethetic dataset similar to the one in the paper.\
    If you want to use custom covariance matrices and means for the multivariate Gaussians,
    use 

    Args:
        dim (int): dimension of samples
            - Dataset 1 (dim=2): 2D
            - Dataset 2 (dim=3): 3D
            - Dataset 3 (dim=4): 4D
        seed (int, optional): numpy RNG seed. Defaults to 42.
            If set to None, then becomes fully random

    Returns:
        Tuple[np.ndarray, np.ndarray]: data, labels
    """
    # change dimension based on the type
    if seed is not None:
        np.random.seed(seed)

    mean_neg = np.zeros(dim)
    mean_pos = np.zeros(dim)
    # along x axis
    mean_neg[0] =  1
    mean_pos[0] = -1

    # take multivariate normal samples (assuming independent covariance)
    class1 = np.random.multivariate_normal(mean_neg, np.eye(dim), 60)
    labels1 = np.zeros(60)  # negative labels '0'

    # Generate data for Class 2 (positive class)
    class2 = np.random.multivariate_normal(mean_pos, np.eye(dim), 100)
    labels2 = np.ones(100)  # positive labels '1'

    # Combine the data
    data = np.vstack((class1, class2))
    labels = np.hstack((labels1, labels2))
    return data, labels

def class_distribution(labels: np.ndarray) -> List[float]:
    """ Returns class label distribution for a set of labels"""
    vals, counts = np.unique(labels, return_counts=True)
    return counts / np.sum(counts)


# plotters # 
def plot_dataset(data: np.ndarray, labels: np.ndarray, projection_dim: int=3, **kwargs) -> None:
    """ Plots classes in 2D/3D. If data is 4+ dimensional, choose 2 or 3D PCA projection """
    if data.shape[1] == 2:
        plot_2d_dataset(data, labels, **kwargs)
    elif data.shape[1] == 3:
        plot_3d_dataset(data, labels, **kwargs)
    else:
        tsne = PCA(n_components=min(projection_dim, 3), random_state=kwargs.get("seed", 0))
        proj_data = tsne.fit_transform(data)
        if projection_dim == 2:
            plot_2d_dataset(proj_data, labels, **kwargs)
        if projection_dim >= 3:
            plot_3d_dataset(proj_data, labels, **kwargs)


def plot_2d_dataset(data, labels, **kwargs):
    dim = data.shape[1]  # only plottable if this is 3D or lower
    assert dim < 4 and dim > 0
    fig = plt.figure(figsize=kwargs.get("figsize", (5,5)))
    ax = fig.add_subplot()

    for i in range(len(np.unique(labels))):
        ax.scatter(
            data[labels==i][:, 0],
            data[labels==i][:, 1],
            marker='+',
            c=COLOR_PALETTE[i],
        )    
    # plt.scatter(data[labels==0][:, 0], data[labels==0][:, 1], marker='+', color='red', label='negative')
    # plt.scatter(data[labels==1][:, 0], data[labels==1][:, 1], marker='+', color='blue', label='positive')
    plt.xlabel(kwargs.get('xlabel', 'x'))
    plt.ylabel(kwargs.get('ylabel', 'y'))
    plt.title(kwargs.get("title", '2D plot'))
    plt.show()

def plot_3d_dataset(data, labels, **kwargs) -> None:
    fig = plt.figure(figsize=kwargs.get("figsize", (6,8)))
    ax = fig.add_subplot(projection='3d')

    colors = ["red", "blue", "green", "pink", "cyan"]
    for i in range(len(np.unique(labels))):
        ax.scatter(
            data[labels==i][:, 0],
            data[labels==i][:, 1],
            data[labels==i][:, 2],
            marker='+',
            c=COLOR_PALETTE[i],
        )

    ax.set_xlabel(kwargs.get('xlabel', 'x'))
    ax.set_ylabel(kwargs.get('ylabel', 'y'))
    ax.set_zlabel(kwargs.get('zlabel', 'z'))
    plt.show()

# evaluation # 
def accuracy_splits(
    X,
    y, 
    classifier, 
    runs=10, 
    test_size=0.3, 
) -> Tuple[float, float, List]:
    """
    Calculate "Leave-One-Out" Cross-Validation accuracy for a classifier.
    NOTE: may need to adjust DKNN to follow the sklearn interface (fit, predict)
    
    Args:
        X (array-like): Feature matrix
        y (array-like): Target vector
        classifier (sklearn estimator): Initialized classifier object
    
    Returns:
        float: LOOCV accuracy mean
        float: LOOCV std
        List:  accuracies per run
    """
    accuracies = []

    rs = ShuffleSplit(n_splits=runs, test_size=test_size)
    for i, (train_index, test_index) in enumerate(rs.split(X)):
        classifier.fit(X[train_index], y[train_index])
        classifier.fit(X[train_index], y[train_index])
        acc = classifier.score(X[test_index], y[test_index])
        accuracies.append(acc)
    # for run in range(runs):
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=test_size
    #     )
    #     classifier.fit(X_train, y_train)
    #     acc = classifier.score(X_test, y_test)
    #     accuracies.append(acc)
    
    return np.mean(accuracies), np.std(accuracies), accuracies

def run_baseline_models(X, y, test_size=0.3, runs=10, seed=0):
    """
    Compare accuracy between SVM, LDA, CART, and KNN
    
    Args:
        X (array-like): Feature matrix
        y (array-like): Target vector
        test_size (float): ratio of test set
        runs (int): num holdout iterations for LOOCV evaluation
        seed (int): random seed
    
    Returns:
        dict: Dictionary of model names and their LOOCV accuracies
    """
    # paper set these at default parameters
    models = {
        "SVM":  SVC(kernel='linear'),
        "LDA":  LinearDiscriminantAnalysis(),
        "CART": DecisionTreeClassifier(random_state=42),
        "KNN":  KNeighborsClassifier(n_neighbors=3)
    }
    
    results = {}
    
    for name, model in models.items():
        acc_mean, acc_std, acc_list = accuracy(X, y, model, test_size=test_size, runs=runs)
        results[name] = acc_mean
        print(f"{name} Mean Accuracy (from {runs} runs): {acc_mean:.4f} (std: {acc_std:.4f})")
    
    return results

# misc metahelpers # 

def enum_string_array(array: np.ndarray):
    """ Use: Given ndarray of class strings, convert to int indices. """
    targets = np.unique(array, return_inverse=True)
    return targets[1]