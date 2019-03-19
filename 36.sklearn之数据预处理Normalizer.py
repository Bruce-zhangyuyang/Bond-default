import pandas as pd
import numpy as np

class diynorm():
    def __init__(self,norm = 'l2'):
        assert norm in ['l1','l2'], 'norm 只能是l1或者l2'
        self.norm = norm
    def get_data(self,data):
        if data.notnull().any().any():
            return data.values, list(data), list(data.index)
        else:
            col = data.columns
            lo_col = data.isnull().any(0)
            col = col[lo_col]
            raise KeyError('第%S列有缺失值'%col.tolist())
    def fit(self,data):
        self.get_data(data)
    def transform(self,data):
        data,col,row = self.get_data(data)
        if self.norm == 'l1':
            data = self.l1(data)
        else:
            data = self.l2(data)
        return pd.DataFrame(data,index=row, columns=col)
    def l1(self,data):
        return data/np.abs(data).sum(1).reshape(-1,1)
    def l2(self,data):
        return data/np.sqrt(np.power(data , 2).sum(1)).reshape(-1,1)
    def fit_transform(self,data):
        return self.transform(data)
a= pd.DataFrame(np.random.randint(-5,5,(4,5)))
norm = diynorm()
norm1 = diynorm(norm='l1')
a = norm.fit_transform(a)
b = norm1.fit_transform(a)