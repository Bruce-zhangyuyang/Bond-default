from sklearn.decomposition import PCA
from sklearn import datasets


#数据
iris = datasets.load_iris()
x = iris.data
y = iris.target

#主成分分析
pca = PCA(n_components=4)
pca.fit(x)
x_new = pca.fit_transform(x)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

pca2 = PCA(n_components=2)
pca2.fit(x)
x_new2 = pca.fit_transform(x)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

from matplotlib import pyplot as plt
plt.scatter(x_new2[:,0],x_new2[:,1] ,marker='o', c=y)#x_new2[:,0]是最大的方向，x_new2[:,1]是次之的方向
plt.show()

