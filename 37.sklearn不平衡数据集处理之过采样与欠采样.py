#过采样
import numpy as np
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x= np.array([[1,1],[1,2],[1,3],[1,3],[1,3],[1,4],[1,6],[2,4],[6,9]])
y = np.array([1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 2])
X_resampled = ros.fit_sample(x,y)
print(X_resampled)


#欠采样
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled2 = rus.fit_sample(x,y)
print(X_resampled2)


#SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled3 = smote.fit_sample(x,y)
