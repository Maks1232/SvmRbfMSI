import numpy as np
import sklearn.svm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn import datasets


class SVM:
    def __init__(self, C=1, gamma=0.00001):
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


# d_set = np.genfromtxt("audit.csv", delimiter=",", skip_header=1)
#
# X = d_set[:, :-2]
# y = d_set[:, -1].astype(int)
#
# df = pd.DataFrame(d_set, columns=['Sector_score','LOCATION_ID','PARA_A','Score_A','Risk_A','PARA_B','Score_B','Risk_B','TOTAL','numbers',
#                                   'Score_B','Risk_C','Money_Value','Score_MV','Risk_D','District_Loss','PROB','RiSk_E','History','Prob',
#                                   'Risk_F','Score','Inherent_Risk','CONTROL_RISK','Detection_Risk','Audit_Risk','Risk'])
#
# df['Risk'] = df['Risk'].apply(int)
# df_2 = df.copy()
# df_2 = df_2.dropna()
# df_2.to_csv('ready_csv.csv', sep=',', index=False)
# d_set = np.genfromtxt("ready_csv.csv", delimiter=",", skip_header=1)
#
# X = d_set[:, :-2]
# y = d_set[:, -1].astype(int)

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
# X, y = datasets.make_moons(n_samples=300, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=259
    )

clf = SVM()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
print("Accuracy (scratch): %.3f" % accuracy)

clf_sklearn = sklearn.svm.SVC(kernel='rbf', gamma=0.00001)
clf_sklearn.fit(X_train, y_train)
y_pred2 = clf_sklearn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred2)
print("Accuracy (Scikit-learn): %.3f" % accuracy)
print(y_pred2)
fig, (ax, ax2) = plt.subplots(2, 2)
fig.tight_layout(pad=2.5)
ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr')

ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='bwr')
ax[1].set_title('Prediction score: %.3f' % metrics.accuracy_score(y_test, y_pred))

ax2[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr')

ax2[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred2, cmap='bwr')
ax2[1].set_title('Prediction score wzorca: %.3f' % metrics.accuracy_score(y_test, y_pred2))

plt.show()
