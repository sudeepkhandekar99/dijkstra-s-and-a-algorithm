#creating testing sets

import pandas as pd
import numpy as np
import random as rd

from tqdm import tqdm
from base import dijkstra

tqdm.pandas()

rd.seed(69)

def get_dj_itr(dataset):
    for idx, row in dataset.iterrows():
        row['iterations_dijkshtra'] = dijkstra(row['input'])

def get_astar_itr(dataset):
    for idx, row in dataset.iterrows():
        pass

dataset = pd.DataFrame(columns=['order', 'input', 'time_a_star', 'time_dijkstra', 'iterations_a_star', 'iterations_dijkstra'])

order_ = 1
all_orders = []
all_inputs = []

for test in range(6):
    order_ += 1
    up_lim = order_*order_

    for loop in range(20):
        y = np.array(rd.sample(range(up_lim),up_lim))
        z = np.reshape(y, (order_,order_))
        z = z.tolist()
        all_orders.append(order_)
        all_inputs.append(z)

dataset['order'] = all_orders
dataset['input'] = all_inputs
dataset['time_a_star'] = None
dataset['time_dijkstra'] = None
dataset['iterations_a_star'] = dataset['input'].progress_apply(dijkstra)
dataset['iterations_dijkstra'] = None #get_astar_itr(dataset)

dataset.to_csv("dataset.csv", index=False)