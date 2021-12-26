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
import gpflow
from gpflow.utilities import ops, print_summary
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
from gpflow.ci_utils import ci_niter
################
# Preprocessing
################

def load_data(dataset):
    if dataset == "H9":
        cell_line = "H9"
        raw_Y = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/Cyclum-master/data/McDavid/h9_df.pkl').T
        cpt = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/Cyclum-master/data/McDavid/h9_cpt.pkl').values
        print("DATASET: ", cell_line)
        print("Original dimesion %d cells x %d genes." % raw_Y.shape)
        print(f"G0/G1 {sum(cpt == 'g0/g1')}, S {sum(cpt == 's')}, G2/M {sum(cpt == 'g2/m')}")
    elif dataset == "mb":
        cell_line = "mb"
        raw_Y = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/Cyclum-master/data/McDavid/mb_df.pkl').T
        cpt = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/Cyclum-master/data/McDavid/mb_cpt.pkl').values
        print("DATASET: ", cell_line)
        print("Original dimesion %d cells x %d genes." % raw_Y.shape)
        print(f"G0/G1 {sum(cpt == 'g0/g1')}, S {sum(cpt == 's')}, G2/M {sum(cpt == 'g2/m')}")
    elif dataset == "pc3":
        cell_line = "pc3"
        raw_Y = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/Cyclum-master/data/McDavid/pc3_df.pkl').T
        cpt = pd.read_pickle('/home/pau/Desktop/MASTER/Statistical_computation/Cyclum-master/data/McDavid/pc3_cpt.pkl').values
        print("DATASET: ", cell_line)
        print("Original dimesion %d cells x %d genes." % raw_Y.shape)
        print(f"G0/G1 {sum(cpt == 'g0/g1')}, S {sum(cpt == 's')}, G2/M {sum(cpt == 'g2/m')}")
    else:
        raise NotImplementedError("Unknown dataset {dataset}")
    
    return raw_Y, cpt
        
# Load data & Transform it to numpy
dataset = "pc3" 
data, cpt = load_data(dataset)
data = data.to_numpy()
data = preprocessing.scale(data)

###############
# MODELS
###############

# Activations
def sinus(x):
  return tf.math.sin(x)

def cosinus(x):
  return tf.math.cos(x)
def linear(x):
  return x
# register activations
get_custom_objects().update({'sinus': Activation(sinus)})
get_custom_objects().update({'cosinus': Activation(cosinus)})
get_custom_objects().update({'linear': Activation(linear)})
#Create 3 branches (1 per activation) for the decoder
encoding_dim = 1
input = keras.Input(shape=(253,))
encoded1 = layers.Dense(1000, activation='tanh')(input)
encoded2 = layers.Dense(253, activation='tanh')(encoded1)
encoded = layers.Dense(encoding_dim, activation='tanh')(encoded2)
x_1 = Activation(sinus, name='sinus')(encoded)
x_2 = Activation(cosinus, name='cosinus')(encoded)
x_3 = Activation(linear, name='linear')(encoded)
output = concatenate([x_1, x_2, x_3])
# branch1 = layers.Dense(253, activation=Activation(sinus, name='sinus'))(encoded)
# branch2 = layers.Dense(253, activation=Activation(cosinus, name='cosinus'))(encoded)
decoded = layers.Dense(253, activation=Activation(linear, name='linear'))(output)
#decoded = concatenate([branch1, branch2, branch3])
autoencoder = keras.Model(input, decoded)
encoder = keras.Model(input, encoded)
encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
x_1 = Activation(sinus, name='sinus')(encoded_input)
x_2 = Activation(cosinus, name='cosinus')(encoded_input)
x_3 = Activation(linear, name='linear')(encoded_input)
output = concatenate([x_1, x_2, x_3])
decoder = keras.Model(encoded_input, decoder_layer(output))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Evaluation
Y_train, Y_test = train_test_split(
    data, test_size=0.33, random_state=0)

Y_train.shape
Y_test.shape

autoencoder.fit(Y_train, Y_train,
                epochs=2000,
                batch_size=256,
                shuffle=True,
                validation_data=(Y_test, Y_test))

encoded = encoder.predict(Y_test)
decoded = decoder.predict(encoded)
#################
# CYCLUM
#################
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


####################
# GPLVM
####################
Y = tf.convert_to_tensor(data, dtype=default_float())
print("Number of points: {} and Number of dimensions: {}".format(Y.shape[0], Y.shape[1]))
latent_dim =   1  # number of latent dimensions
num_inducing = 10 # number of inducing pts
num_data = Y.shape[0]  # number of data points
X_mean_init = ops.pca_reduce(Y, latent_dim)
X_var_init = tf.ones((num_data, latent_dim), dtype=default_float())
inducing_variable = tf.convert_to_tensor(
    np.random.permutation(X_mean_init.numpy())[:num_inducing], dtype=default_float()
)
#Kernels
lengthscales = tf.convert_to_tensor([1.0] * latent_dim, dtype=default_float())
var = tf.convert_to_tensor([1.0] * latent_dim, dtype=default_float())
k1 = gpflow.kernels.RBF(lengthscales=lengthscales)
k2 = gpflow.kernels.Matern32(lengthscales=lengthscales)
k3_base = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
k3 = gpflow.kernels.Periodic(k3_base, period=1.0)
k = k1+k2+k3 + gpflow.kernels.White(variance=0.01)
k = k1 + gpflow.kernels.White(variance=0.05)

gplvm = gpflow.models.BayesianGPLVM(
    Y,
    X_data_mean=X_mean_init,
    X_data_var=X_var_init,
    kernel=k,
    inducing_variable=inducing_variable,
)
gplvm.likelihood.variance.assign(0.01)
opt = gpflow.optimizers.Scipy()
maxiter = ci_niter(3000)
opt.minimize(
    gplvm.training_loss,
    method="BFGS",
    variables=gplvm.trainable_variables,
    options=dict(maxiter=maxiter),
)
print_summary(gplvm)
pseudotime = gplvm.X_data_mean.numpy()

