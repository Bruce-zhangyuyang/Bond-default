from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets

iris = datasets.load_iris()
data = iris['data'][:, (2, 3)] #petal length pental width
y = (iris['target'] == 2).astype(np.float64)
svm_clf = Pipeline((('scaler',StandardScaler()),
     ('linear_svc',LinearSVC(loss='hinge')),
))

svm_clf.fit(data,y)
print(svm_clf.predict([[5.5,1.7]]))


#SGDClassifier
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#构造球形数据集
(x,y) = make_moons(200, noise=0.2)


'''
非线性SVM分类(Nonlinear SVM CLassification)
多项式核(Polynomial Kernal)
'''
poly_kernal_svm_clf = Pipeline((
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
))
#其中参数coef0为高degree特征相比低degree特征对模型的影响程度，参数degree为选择多项式特征的程度，参数C为松弛因子
poly_kernal_svm_clf.fit(x,y)
#画出决策画

import numpy as np
from matplotlib import pyplot as plt
xx, yy = np.meshgrid(np.arange(-2, 3, 0.01), np.arange(-1, 2, 0.01))
y_new = poly_kernal_svm_clf.predict(np.c_[xx.ravel(),yy.ravel()])
plt.contourf(xx, yy, y_new.reshape(xx.shape), cmap='PuBu')
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
plt.show()


'''
高斯径向基核 (Gaussian RBF Kernel)
'''

rbf_kernel_svm_clf = Pipeline((
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
))
'''
如果gamma比较大，会使得决策线变窄，变得不规则。相反，小的gamma使决策线变宽，变平滑。
所以就像一个正则化参数：如果你的模型过拟合，可以适当减少它，如果它欠拟合，可以增加它（类似于C超参数）
'''
rbf_kernel_svm_clf.fit(x, y)
#画出决策线

import numpy as np
from matplotlib import pyplot as plt
xx, yy = np.meshgrid(np.arange(-2, 3, 0.01), np.arange(-1, 2, 0.01))
y_new = rbf_kernel_svm_clf.predict(np.c_[xx.ravel(),yy.ravel()])
plt.contourf(xx, yy, y_new.reshape(xx.shape), cmap='PuBu')
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
plt.show()


ss = StandardScaler()
x_ss = ss.fit_transform(x)
svclist = [
    SVC(kernel='rbf', gamma=1, C=0.001), SVC(kernel='rbf', gamma=1, C=10),
    SVC(kernel='rbf', gamma=10, C=10), SVC(kernel='rbf', gamma=10, C=10)
]
#画出决策图
xx, yy = np.meshgrid(np.arange(-2, 3, 0.001),np.arange(-1, 2, 0.01))
fig, axes = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        model = svclist[2*i+j]
        model.fit(x_ss, y.ravel())
        y_new = model.predict(np.c_[xx.ravel(), yy.ravel()])
        axes[i][j].contourf(xx, yy, y_new.reshape(xx.shape), cmap='PuBu')
        axes[i][j].scatter(x[:, 0], x[:, 1], marker='o', c=y)
plt.show()