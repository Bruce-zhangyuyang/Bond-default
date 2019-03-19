from sklearn import datasets
digits = datasets.load_digits()

X = digits.data[:500]
Y = digits.target[:500]

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X)

from matplotlib import pyplot as plt
plt.figure(figsize=(13,8))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i , c , l in zip(range(len(digits.target_names)) , colors , digits.target_names ):
     plt.scatter(X_2d[Y==i,0],X_2d[Y==i,1],c=c,label = l)
plt.legend()
plt.show()