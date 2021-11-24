import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import quantecon as qe
from quantecon.distributions import BetaBinomial
from numba import jit, float64, njit, prange
from numba.experimental import jitclass
plt.rcParams["figure.figsize"] = (11, 5)

### Largely inspired by quantecon ###

##################
### Parameters ###
##################

unemployment_salary = 2
β = 0.90

application_cost = 0.5

γ = 0.5

n = 300                                 
w_min, w_max = 2, 5
w_default = np.linspace(w_min, w_max, n+1)      
a, b = 2.5, 2
q_default = BetaBinomial(n, a, b).pdf()


@njit
def u(c, σ=1.3):
    return (c**(1 - σ) - 1) / (1 - σ)
  
mccall_data = [
    ('α', float64),      # job separation rate
    ('β', float64),      # discount factor
    ('c', float64),      # unemployment compensation
    ('w', float64[:]),   # list of wage values
    ('p', float64[:]),    # pmf of random variable w
    ('γ', float64),
    ('z', float64)
]

@jitclass(mccall_data)
class McCallModel:
    """
    Stores the parameters and functions associated with a given model.
    """

    def __init__(self, γ=γ, α=0.0, β=β, c=unemployment_salary, w=w_default, p=q_default, z=application_cost):

        self.α, self.β, self.c, self.w, self.p, self.γ, self.z = α, β, c, w, p, γ, z


    def update(self, s, q):

        α, β, c, w, p, γ, z = self.α, self.β, self.c, self.w, self.p, self.γ, self.z

        s_new = np.empty_like(s)
        q_new = np.empty_like(q)
        
        for j in range(len(w)):
            prod = w[j]
            d_down = np.sum(p * s[:, max(j-1, 0)])
            d_same = np.sum(p * s[:, j])
            for i in range(len(w)):
                wage = w[i]
                s_new[i, j] = np.max(np.array([(prod >= wage) * γ * (u(wage - z) + β * q[i, j]) + (prod >= wage) * (1 - γ) * (u(c - z) + β * d_down) + (prod < wage) * (u(c - z) + β * d_down),
                                               u(c) + β * d_down]))
                q_new[i, j] = np.max(np.array([(prod >= wage) * (u(wage) + β * q[i, min(j+1, n)]) + (prod < wage) * (u(c) + β * d_same),
                                               u(c) + β * d_same]))

        return s_new, q_new
      
      
@njit
def solve_model(mcm, tol=1e-5, max_iter=2000):
    """
    Iterates to convergence on the Bellman equations

    * mcm is an instance of McCallModel
    """

    s = np.ones((n+1, n+1))    # Initial guess of s
    q = np.ones((n+1, n+1))    # Initial guess of q
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        s_new, q_new = mcm.update(s, q)
        error_1 = np.max(np.abs(s_new - s))
        error_2 = np.max(np.abs(q_new - q))
        error = max(error_1, error_2)
        s = s_new
        q = q_new
        i += 1
 
    if i == max_iter:
        print('Max iteration reached')
    return s, q
  
mcm = McCallModel(γ=0.5, c=unemployment_salary)
s, q = solve_model(mcm, tol=1e-5, max_iter=2000)

l_q = []
l_s = []
for i in range(n+1):
    q_0 = u(c) + gamma * np.sum(q_default * s[:, i])
    l_q.append(w_default[np.argmax(q[:, i] > q_0)])
    s_0 = u(c) + gamma * np.sum(q_default * s[:, max(i-1,0)])
    l_s.append(w_default[np.argmax(s[:, i] > s_0)])
