from utils.popphy_io import get_config
import datetime
import time
import pandas as pd
import  numpy as np
cv_list=[0,1,2,3]
stat_df = pd.DataFrame(index=["AUC", "Accuracy","F1"], columns=cv_list)
# stat_df=pd.DataFrame({"AUC":[12, 4, 5, 44, 1],
#                    "B":[5, 2, 54, 3, 2],
#                    "C":[20, 16, 7, 3, 8],
#                    "D":[14, 3, 17, 2, 6]})

stat_df.loc["AUC"]=[12, 4, 5, 44]
stat_df.loc["Accuracy"]=[5, 2, 54, 3]
stat_df.loc["F1"]=[20, 16, 7, 3]
vector =[20, 16, 7, 3]
hmatrix = np.array(vector)
result=dict(stat_df.mean(1))
print(result)
print(hmatrix)

normaladj=np.array([[20, 16, 7],[20, 16, 7],[12, 4, 5]])
print(normaladj.shape)
a=np.zeros(3)
a[0]=1
a[1]=2
a[2]=3
print(a.shape)
outmetrix = np.array([1,2,3])
print(outmetrix.shape)
dadhmaps = np.matmul(normaladj,outmetrix)
badhmaps = np.matmul(normaladj,a)
print(dadhmaps)
print(badhmaps)

train_x=np.expand_dims(normaladj,-1)
print(train_x.shape)
