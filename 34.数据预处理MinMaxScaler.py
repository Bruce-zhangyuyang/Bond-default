import numpy as np
import pandas as pd

class diymm():
    '接受pdDataFrame数据'
    def __init__(self,under = 0, up = 1):
        assert under != up , 'under 和 up不能相等'
        self.under = under
        self.up = up
        self.max_list = None
        self.min_list = None
    def get_data(self,data):
        if data.notnull().all().all():
            return data.values , list(data) , list(data.index)
        else:
            lose_col = data.columns
            col_lo = data.isnull().any(0)
            lose_col = lose_col[col_lo]
            raise KeyError('第%s列有空值'%lose_col.tolist())
    def fit(self,data):
        data , _ , _ = self.get_data(data)
        self.max_list = data.max(0)
        self.min_list = data.min(0)
    def transform(self,data):
        if self.max_list is None:
            raise KeyError('没有经过fit，请先fit')
        else:
            data , col , row = self.get_data(data)
            data = data - self.min_list.reshape(1,-1)
            data = data/(self.max_list.reshape(1,-1)-self.min_list.reshape(1,-1))
            if self.under ==0 and self.up == 1:
                pass
            else:
                data = data*(self.up-self.under)+self.under
            return pd.DataFrame(data , index=row, columns= col)
    def fit_transform(self,data):
        if self.max_list is None:
            self.fit(data)
        return self.transform(data)
s = pd.DataFrame(np.random.randint(0,10,(4,5)))
mm = diymm()
mm.fit_transform(s)
