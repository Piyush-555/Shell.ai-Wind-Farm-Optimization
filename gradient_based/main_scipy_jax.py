import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
import pandas as pd
import scipy
from scipy import optimize
import matplotlib.pyplot as plt

import evaluator_jax as evaluator

iteration = 0

def get_constraints_eqn():
    constraints = []
    for i in range(50):
        for j in range(i+1, 50):
            def constraint(params):
                turb_params = params.reshape((50, 2))
                return jnp.sqrt((turb_params[i][0] - turb_params[j][0])**2 + (turb_params[i][1] - turb_params[j][1])**2 + 1e-6) - 0.1
            grad_constraint = jax.grad(constraint, 0)
            constraints.append({'type': 'ineq', 'fun': constraint, 'jac': grad_constraint})
    return constraints


def objective_eqn(params):
    power_curve = evaluator.loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')
    power_curve_jax = jnp.array(power_curve)
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = evaluator.preProcessing(power_curve)

    turb_coords = param2coords(params)
    turb_rad = 50

    wind_dir = "Shell_Hackathon Dataset/Wind Data"
    wind_inst_freq = evaluator.binWindResourceData(os.path.join(wind_dir, "all.csv"))

    aep = evaluator.getAEP(turb_rad, turb_coords, power_curve_jax, wind_inst_freq, 
                    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
    global iteration
    iteration += 1
    print("Iteration:", iteration, "\tAEP:", aep)

    # save_params(params, "jax_tcv0" + str(np.array(aep).item()) + ".csv")

    return -aep

# def grad_objective_eqn(params):
#     return np.array(jax.grad(objective_eqn)(params))

grad_objective_eqn = jax.grad(objective_eqn, 0)
# grad_grad_objective_eqn = jax.hessian(objective_eqn, 0)


def coords2param(turb_coords):
    return turb_coords.flatten() / 4000


def param2coords(params):
    return params.reshape((50, 2)) * 4000


def save_params(params, filename):
    coords = param2coords(params)
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1]
    })
    df.to_csv('Submissions/' + filename, index=False)


if __name__ == "__main__":
    num_iter = 10

    path = "Submissions/original.csv"
    turb_coords = evaluator.getTurbLoc(path)
    turb_coords = jnp.array(turb_coords)

    params0 = coords2param(turb_coords)
    bounds = jnp.array([(0.0125, 0.9875)] * 100)
    cnstr = get_constraints_eqn()
    options = {
        'disp': True,
        'maxiter': num_iter
    }
    # import pdb; pdb.set_trace()

    result = optimize.minimize(objective_eqn, params0, jac=grad_objective_eqn, method='trust-constr', bounds=bounds, constraints=cnstr, options=options)
    print(result)
    save_params(result.x, 'jax_tcv0.csv')
    coords = param2coords(result.x)
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.show()

