import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cyclum.tuning
import keras
from keras import layers
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import concatenate
from sklearn.model_selection import train_test_split
import cyclum.models
import cyclum.evaluation
################
# Preprocessing
################

def load_data(dataset):
    if dataset == "H9":
        cell_line = "H9"
        raw_Y = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/project4/data/McDavid/h9_df.pkl').T
        cpt = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/project4/data/McDavid/h9_cpt.pkl').values
        print("DATASET: ", cell_line)
        print("Original dimesion %d cells x %d genes." % raw_Y.shape)
        print(f"G0/G1 {sum(cpt == 'g0/g1')}, S {sum(cpt == 's')}, G2/M {sum(cpt == 'g2/m')}")
    elif dataset == "mb":
        cell_line = "mb"
        raw_Y = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/project4/data/McDavid/mb_df.pkl').T
        cpt = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/project4/data/McDavid/mb_cpt.pkl').values
        print("DATASET: ", cell_line)
        print("Original dimesion %d cells x %d genes." % raw_Y.shape)
        print(f"G0/G1 {sum(cpt == 'g0/g1')}, S {sum(cpt == 's')}, G2/M {sum(cpt == 'g2/m')}")
    elif dataset == "pc3":
        cell_line = "pc3"
        raw_Y = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/project4/data/McDavid/pc3_df.pkl').T
        cpt = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/project4/data/McDavid/pc3_cpt.pkl').values
        print("DATASET: ", cell_line)
        print("Original dimesion %d cells x %d genes." % raw_Y.shape)
        print(f"G0/G1 {sum(cpt == 'g0/g1')}, S {sum(cpt == 's')}, G2/M {sum(cpt == 'g2/m')}")
    else:
        raise NotImplementedError("Unknown dataset {dataset}")
    
    return raw_Y, cpt
        
# Load data & Transform it to numpy
dataset = "H9" 
data, cpt = load_data(dataset)
data = data.to_numpy()
data = preprocessing.scale(data)

# Train (H9)
model = cyclum.tuning.CyclumAutoTune(data, max_linear_dims=3, 
                                     epochs=2000, rate=5e-4, verbose=100,
                                     encoder_width=[30, 20])

model = cyclum.models.AutoEncoder(input_width=data.shape[1],
                                  encoder_width=[30, 20], 
                                  encoder_depth=2,
                                 n_circular_unit=2,
                                 n_logistic_unit=0,
                                 n_linear_unit=0,
                                 n_linear_bypass=3,
                                 dropout_rate=0.1)

model.train(data, epochs=1000, verbose=100, rate=2e-4)
pseudotime = model.predict_pseudotime(data)
flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score Cyclum: ", dis_score)

# MB
dataset = "mb" 
data, cpt = load_data(dataset)
data = data.to_numpy()
data = preprocessing.scale(data)
model = cyclum.tuning.CyclumAutoTune(data, max_linear_dims=5, 
                                     epochs=2000, rate=5e-4, verbose=100,
                                      encoder_width=[30, 20])

model.train(data, epochs=1000, verbose=100, rate=2e-4)
pseudotime = model.predict_pseudotime(data)
flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score Cyclum: ", dis_score)

# PC3
dataset = "pc3" 
data, cpt = load_data(dataset)
data = data.to_numpy()
data = preprocessing.scale(data)
# model = cyclum.tuning.CyclumAutoTune(data, max_linear_dims=5, 
#                                       epochs=2000, rate=5e-4, verbose=100,
#                                       encoder_width=[30, 20])

model = cyclum.models.AutoEncoder(input_width=data.shape[1],
                                  encoder_width=[30, 20], 
                                  encoder_depth=2,
                                 n_circular_unit=2,
                                 n_logistic_unit=0,
                                 n_linear_unit=0,
                                 n_linear_bypass=3,
                                 dropout_rate=0.1)
model.train(data, epochs=1000, verbose=100, rate=2e-4)

pseudotime = model.predict_pseudotime(data)
flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score Cyclum: ", dis_score)

import cyclum.illustration
color_map = {"g0/g1": "red", "s": "green", "g2/m": "blue"}
cyclum.illustration.plot_round_distr_color(pseudotime[:, 0], cpt.squeeze(), color_map)


#####################
# CHLA 9
##################
data = pd.read_csv("/home/pau/Desktop/MASTER/Statistical_computation/project4/final_dataset.csv")
data = data.iloc[: , 1:]
data = data.to_numpy()
data = preprocessing.scale(data)

model = cyclum.models.AutoEncoder(input_width=data.shape[1],
                                  encoder_width=[600, 300, 200, 100, 50, 20], 
                                  encoder_depth=6,
                                  n_circular_unit=2,
                                  n_logistic_unit=0,
                                  n_linear_unit=5,
                                  n_linear_bypass=3,
                                  dropout_rate=0.1)
model.train(data, epochs=20, verbose=10, rate=2e-4)


pseudotime = model.predict_pseudotime(data)
pseudotime = (pseudotime % (2 * np.pi)) / 2
pseudotime = pseudotime[:, 0]
label = np.array([0 for p in pseudotime if p < 1.5] )
import seaborn as sns
sns.distplot(pseudotime, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = pseudotime)
