from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

iris = load_iris()
target = iris['target']
ss = StandardScaler()
iris = pd.DataFrame(ss.fit_transform(iris['data']),columns = iris['feature_names'])
x_0 , x_1 , y_0 , y_1 =train_test_split(iris , target, random_state=0)
lr_1 = LogisticRegression()
lr_2 = LogisticRegression(penalty='l1')

lr_1.fit(x_0,y_0)
lr_2.fit(x_0,y_0)

lr_11 = lr_1.coef_
lr_22 = lr_2.coef_
#在L2中, 系数会相对平滑,  而在L1中, 不重要的特征的系数会变成0,
#  所以说L1的系数会相对极端一点.
# 正是因为这个特性, L1的逻辑回归常用来筛选特征.


