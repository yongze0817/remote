import json
import numpy as np
import pandas as pd
with open('./results/rbf/simu_A_w_100_2000_f.json', 'r') as f:
    data_str = f.read()

results = json.loads(data_str)

for c_uppers, models in results.items():
    print(len(models))
    # print(235/len(models)*(100-len(models)))
