import numpy as np
import sklearn.svm
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tabulate import tabulate
from sklearn import datasets
from scipy import stats


def gamma_scale(X):
    return 1 / (X.shape[1] * X.var())


class SVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, gamma=0.0001):
        self.kernel = lambda x, y: np.exp(-gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1))
        self.C = C

    def fit(self, X, y):

        self.X = X
        self.y = y * 2 - 1  # sprowadzenie etykiet z wartości 0,1 na wartości 1, -1
        self.lambdas = np.zeros_like(self.y, dtype=float)  # uzupelnienie macierzy na wzor wektora etykiet
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


X, y = datasets.make_circles(n_samples=300, random_state=1410)

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
rskf.get_n_splits(X, y)

clfs = {
    'SVM (scratch)': SVM(gamma=gamma_scale(X)),
    'SVM (rbf sklearn)': sklearn.svm.SVC(kernel='rbf', gamma='scale'),
    'SVM (linear sklearn)': sklearn.svm.SVC(kernel='linear'),
    'SVM (sigmoid sklearn)': sklearn.svm.SVC(kernel='sigmoid', gamma='scale'),
    'Logistic Regression': sklearn.linear_model.LogisticRegression(max_iter=5000),
}
acc_scores = np.zeros(shape=[len(clfs), rskf.get_n_splits()])
f1_scores = np.zeros(shape=[len(clfs), rskf.get_n_splits()])

fig, axs = plt.subplots(5, 2, figsize=(18, 18))
fig.suptitle("Real dataset binary classification", fontsize=30)
fig.tight_layout(pad=5)

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, (clf_name, clf) in enumerate(clfs.items()):

        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        acc_scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
        f1_scores[clf_id, fold_id] = sklearn.metrics.f1_score(y[test], y_pred)

        axs[clf_id, 0].scatter(X[test][:, 0], X[test][:, 1], c=y[test], cmap='bwr')
        axs[clf_id, 1].scatter(X[test][:, 0], X[test][:, 1], c=y_pred, cmap='bwr')
        axs[clf_id, 0].set_title(clf_name + " test", fontsize=18)
        axs[clf_id, 1].set_title(clf_name + " prediction", fontsize=18)

mean_scores = np.mean(acc_scores, axis=1)
std_scores = np.std(acc_scores, axis=1)
mean_f1_scores = np.mean(f1_scores, axis=1)

np.save('accuracy_results.npy', acc_scores)
np.save('f1scores_results.npy', f1_scores)

metrics = {
    "ACCURACY": acc_scores,
    "F1_SCORE": f1_scores
}

# Testy statystyczne
for metric_id, (metric_name, metric) in enumerate(metrics.items()):

    means_vector = np.mean(metric, axis=1)

    t_stats = np.zeros((len(clfs), len(clfs)))
    p = np.zeros(t_stats.shape)
    p_bools = np.zeros(t_stats.shape, dtype=bool)
    p_alpha = np.zeros(p_bools.shape, dtype=bool)
    s_domination = np.zeros(p_bools.shape, dtype=bool)

    for clfs_y in range(len(clfs)):
        for clfs_x in range(len(clfs)):
            t_stats[clfs_y, clfs_x], p[clfs_y, clfs_x] = stats.ttest_rel(metric[clfs_y, :], metric[clfs_x, :])
            if means_vector[clfs_y] > means_vector[clfs_x]:
                p_bools[clfs_y, clfs_x] = True
            if p[clfs_y, clfs_x] < .05:
                p_alpha[clfs_y, clfs_x] = True

    s_domination = p_bools * p_alpha

    print("\nt_stats:")
    print(t_stats)
    print("-----------------------")
    print("\np_values:")
    print(p)
    print("-----------------------")
    print("\np_bools:")
    print(p_bools)
    print("-----------------------")
    print("\np_alpha:")
    print(p_alpha)
    print("-----------------------")
    print("\ns_domination:")
    print(s_domination)
    print("-----------------------")

    print(f"\nWyniki testu t-studenta dla metryki {metric_name}:\n")

    for i in range(means_vector.shape[0]):
        for j in range(means_vector.shape[0]):
            if s_domination[i][j]:
                print(list(clfs.keys())[i], "with: %.3f" %means_vector[i], "better than", list(clfs.keys())[j], "with: %.3f" %means_vector[j])
    print()

output_table = [["Classificator", "Mean accuracy", "Standard Deviation", "Mean F1 score"],
                ["SVM (scratch)", mean_scores[0], std_scores[0], mean_f1_scores[0]],
                ["SVM (rbf sklearn)", mean_scores[1], std_scores[1], mean_f1_scores[1]],
                ["SVM (linear sklearn)", mean_scores[2], std_scores[2], mean_f1_scores[2]],
                ["SVM (sigmoid sklearn)", mean_scores[3], std_scores[3], mean_f1_scores[3]],
                ["Logistic Regression", mean_scores[4], std_scores[4], mean_f1_scores[4]]
                ]

print(tabulate(output_table, headers="firstrow", floatfmt=".3f"))

with open('Wyniki/Real_table_gamma_scale.txt', 'w') as f:
    f.write(tabulate(output_table, headers="firstrow", floatfmt=".3f"))

plt.savefig("Wyniki/Real_table_gamma_scale.png")
plt.show()

