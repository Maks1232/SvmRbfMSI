import numpy
import numpy as np
import sklearn.svm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn import metrics
from sklearn import datasets
import pandas as pd
from tabulate import tabulate

class SVM:
    def __init__(self, C=1, gamma=0.0001):
        self.kernel = lambda x, y: np.exp(-gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1))
        self.C = C

    def fit(self, X, y):
        self.X = X
        self.y = y * 2 - 1     # sprowadzenie etykiet z wartości 0,1 na wartości 1, -1
        self.lambdas = np.zeros_like(self.y, dtype=float)   # uzupelnienie macierzy na wzor wektora etykiet
        # self.K - macierz jądrowa, która reprezentuje podobieństwo między każdą parą przykładów treningowych
        self.K = self.kernel(self.X, self.X) * self.y[:, np.newaxis] * self.y

        # Q - macierz Grama, która jest wyznaczana na podstawie wektora cech danych wejściowych oraz funkcji jądra SVM
        # t_max - odległość między hiperpłaszczyzną a najbliższym punktem należącym do jednej z klas danych (margines)
        # v0 - wartość początkowa lambd
        # k0 - wektor wag
        # u - wektor różnicy między wartościami funkcji decyzyjnej dla dwóch punktów

        # self.lambdas - wektor mnożników Lagrange'a
        # b - parametr przesuniecia

        for i in range(len(self.lambdas)):
            for j in range(len(self.lambdas)):
                Q = self.K[[[i, i], [j, j]], [[i, j], [i, j]]]
                v0 = self.lambdas[[i, j]]
                k0 = 1 - np.sum(self.lambdas * self.K[[i, j]], axis=1)
                u = np.array([-self.y[j], self.y[i]])
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-7)
                self.lambdas[[i, j]] = v0 + u * t_max

        # szukanie indeksów próbek ze zbioru treningowego, które mają niezerowe wartości mnożników Lagrange'a i większe niż 1E-7
        idx = np.nonzero(self.lambdas > 1E-7)

        #wyznaczenie b za pomocą tzw. warunków KKT (Karush-Kuhn-Tucker)
        self.b = np.mean((1.0 - np.sum(self.K[idx] * self.lambdas, axis=1)) * self.y[idx])

        return self

    def predict(self, X):
        decision = np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b
        return np.where(np.sign(decision) >= 0, 1, 0)


d_set = pd.read_csv("wdbc.csv", header=None)
d_set.replace({'M': 0, 'B': 1}, inplace=True)

X = d_set.drop([0, 1], axis=1).astype(float)
y = d_set.get(1)

X = X.to_numpy()
y = y.to_numpy()

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
rskf.get_n_splits(X, y)

# mean = []
# mean_2 = []
clfs = {
    'SVM (scratch)': SVM(),
    'SVM (rbf sklearn)': sklearn.svm.SVC(kernel='rbf', gamma=0.0001),
    'SVM (linear sklearn)': sklearn.svm.SVC(kernel='linear'),
    'SVM (sigmoid sklearn)': sklearn.svm.SVC(kernel='sigmoid'),
    'Logistic Regression': sklearn.linear_model.LogisticRegression(max_iter=5000),
}
acc_scores = np.zeros(shape=[len(clfs), rskf.get_n_splits()])
f1_scores = np.zeros(shape=[len(clfs), rskf.get_n_splits()])

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, (clf_name, clf) in enumerate(clfs.items()):

        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        acc_scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
        f1_scores[clf_id, fold_id] = sklearn.metrics.f1_score(y[test], y_pred)

mean_scores = np.mean(acc_scores, axis=1)
std_scores = np.std(acc_scores, axis=1)
f1_scores = np.mean(f1_scores, axis=1)

# for clf_id, clf_name in enumerate(clfs):
#     print(list(clfs.keys())[clf_id]+":", "\t\tMean accuracy score: %.3f" %mean_scores[clf_id], "\tStd accuracy: %.3f" %std_scores[clf_id], "\t\tF1 score: %.3f" %f1_scores[clf_id])

output_table = [["Classificator", "Mean accuracy", "Standardard Deviation", "Mean F1 score"],
                ["SVM (scratch)", mean_scores[0], std_scores[0], f1_scores[0]],
                ["SVM (rbf sklearn)", mean_scores[1], std_scores[1], f1_scores[1]],
                ["SVM (linear sklearn)", mean_scores[2], std_scores[2], f1_scores[2]],
                ["SVM (sigmoid sklearn)", mean_scores[3], std_scores[3], f1_scores[3]],
                ["Logistic Regression", mean_scores[4], std_scores[4], f1_scores[4]]
                ]

print(tabulate(output_table, headers="firstrow", tablefmt="fancy_grid", floatfmt=(".3f")))

    # print(10-i)
    # if i == 9:
    #     print()
    #     print('\x1b[38;2;255;0;0m\x1b[48;2;244;244;0m' + '\/\/\/ Nikt się tego nie spodziewał \/\/\/' + '\x1b[0m')
    #     fig, (ax, ax2) = plt.subplots(2, 2)
    #     fig.tight_layout(pad=2.5)
    #     ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr')
    #
    #     ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='bwr')
    #     ax[1].set_title('Avg prediction score: %.3f' % np.mean(mean))
    #
    #     ax2[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr')
    #
    #     ax2[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred2, cmap='bwr')
    #     ax2[1].set_title('Avg prediction score SKlearn: %.3f' % np.mean(mean_2))
    #
    #     plt.savefig('plot.png')


