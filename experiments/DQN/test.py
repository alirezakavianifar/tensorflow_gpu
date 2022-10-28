import sys
sys.path.append(r'D:\projects\tensorflow_gpu\experiments')
from network import DeepQNetwork
from utils import load_data, packetloss_threshold, energyconsumption_threshold
import pandas as pd
import numpy as np

path = r'D:\projects\papers\Deep Learning for Effective and Efficient  Reduction of Large Adaptation Spaces in Self-Adaptive Systems\experiment_results\simulation_results\DeltaIoTv1'

data = load_data(path=path, load_all=True)

# X = data.iloc[:,:1]
# columns = data.columns
max_val_en = data['energyconsumption'].max()
min_val_en = data['energyconsumption'].min()
max_val_pl = data['packetloss'].max()
min_val_pl = data['packetloss'].min()

data['energyconsumptionThreshold'] = data['energyconsumption'].apply(lambda x: energyconsumption_threshold(x))
data['packetlossThreshold'] = data['packetloss'].apply(lambda x: packetloss_threshold(x))


data['features'] = data['features'].values.tolist()

sampled_data = data.sample(n=5000)

filtered_data = sampled_data.loc[(sampled_data['energyconsumptionThreshold']==1) & (sampled_data['packetlossThreshold']==1)]

# data = data.iloc[:1000,:]

X = sampled_data['features']
X = pd.DataFrame(X)
tt=X.iloc[:1,:]

X = X.values.tolist()

r = np.array(X[0])

lst = []
for ls in X:
    item = pd.DataFrame(ls).T
    lst.append(item)
    
final_X = pd.concat(lst)
final_y = sampled_data['energyconsumptionThreshold']

model = DeepQNetwork(len(final_X.columns), 2, 1e-3)

model.fit(final_X, final_y)

    







