from matplotlib import pyplot as plt
import torch
import pandas as pd
from torch.nn import Parameter
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from gpytorch.priors import NormalPrior
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood

# MODEL

class bGPLVM(ApproximateGP):
    def __init__(self, n_inducing_points, n_latent_dims, n_data_points, n_data_dims, X_prior_mean):
        batch_shape = torch.Size([n_data_dims])
        inducing_points = torch.randn(n_data_dims, n_inducing_points, n_latent_dims)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=True
        )
        super(bGPLVM, self).__init__(variational_strategy)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            gpytorch.kernels.PeriodicKernel(batch_shape=batch_shape),
            batch_shape=batch_shape
        )
        # self.covar_module = ScaleKernel(
        #     RBFKernel(nu=1.5,batch_shape=batch_shape, ard_num_dims=2),
        #     batch_shape=batch_shape
        # )
        self.X = Parameter(X_prior_mean.clone())
        self.register_parameter(
            name="X", 
            parameter=self.X
            )
        self.register_prior('prior_X', NormalPrior(X_prior_mean,torch.ones_like(X_prior_mean)), 'X')
        
    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

# DATA LOADING
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

# selecting the dat & setting the prior
dataset = "mb" 
data, cpt = load_data(dataset)
data = data.to_numpy()
data = preprocessing.scale(data)
data = torch.tensor(data, dtype=torch.get_default_dtype())
prior = cpt
prior = np.concatenate(prior).astype(str)
prior = [p.replace('g0/g1', '1') for p in prior]
prior = [p.replace('s', '2') for p in prior]
prior = [p.replace('g2/m', '3') for p in prior]
prior = np.array([int(p) for p in prior])
np.random.seed(10)
sigma_t = .5
x_mean = [prior[i] + sigma_t * np.random.randn(1) for i in range(0, data.shape[0])] # initialisation of latent_mean 
x_mean = torch.tensor(x_mean, dtype=torch.get_default_dtype())

# training
n_latent_dims = 1
model = bGPLVM(n_inducing_points=32, 
               n_latent_dims=n_latent_dims, 
               n_data_points = data.shape[0], 
               n_data_dims = data.shape[1],
               X_prior_mean = x_mean)

likelihood = GaussianLikelihood(num_tasks=data.shape[1], batch_shape=torch.Size([data.shape[1]]))
model.train()
likelihood.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

mll = PredictiveLogLikelihood(likelihood, model, num_data=data.size(0))

loss_list = []
iterator = trange(500)
for i in iterator:
    optimizer.zero_grad()
    output = model(model.X)
    loss = -mll(output, data.T).sum()
    loss_list.append(loss.item())
    print(str(loss.item()) + ", iter no: " + str(i))
    iterator.set_postfix(loss=loss.item())
    loss.backward(retain_graph=True)
    optimizer.step()
        
pseudotime = model.X.detach().numpy()
flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score: ", dis_score)