import os
import numpy as onp
import jax
import jax.numpy as jnp
from jax import grad
from jax.experimental import optimizers
import pandas as pd
import scipy
from scipy import optimize
import matplotlib.pyplot as plt

import evaluator_jax as evaluator


def get_constraints(params, lambdas):
    l = 0
    turb_params = params.reshape((50, 2))
    constraint_term = 0.0
    
    # Proximity
    for i in range(50):
        for j in range(i+1, 50):
            constraint_term += lambdas[l] * (jnp.sqrt((turb_params[i][0] - turb_params[j][0])**2 + (turb_params[i][1] - turb_params[j][1])**2 + 1e-6) - 0.1)
            l += 1
    
    # Perimeter
    lb = 0.0125
    ub = 0.9875
    for p in range(100):
        constraint_term += lambdas[l] * (params[p] - lb)
        l += 1
        constraint_term += lambdas[l] * (ub - params[p])
        l += 1

    return constraint_term


def lagrangian(params, lambdas, not_grad=False):
    turb_coords = param2coords(params)
    aep = evaluator.getAEP(turb_rad, turb_coords, power_curve_jax, wind_inst_freq, 
                    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)

    constraint_term = get_constraints(params, lambdas)
    lagrange = (-1) * aep - constraint_term
    if not_grad:
        print("AEP:", aep, "\tConstraint_Satisfaction:", constraint_term)

    return lagrange


grad_params_lambdas = jax.grad(lagrangian, argnums=(0, 1))


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
    lambda_init_multiplier = 1
    lambda_grad_multiplier = 1
    learning_rate = 1e-3
    num_steps = 10

    power_curve = evaluator.loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')
    power_curve_jax = jnp.array(power_curve)
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = evaluator.preProcessing(power_curve)
    turb_rad = 50

    wind_dir = "Shell_Hackathon Dataset/Wind Data"
    wind_inst_freq = evaluator.binWindResourceData(os.path.join(wind_dir, "all.csv"))

    path = "Submissions/original.csv"
    turb_coords = evaluator.getTurbLoc(path)
    turb_coords = jnp.array(turb_coords)

    params0 = coords2param(turb_coords)
    lambdas0 = jnp.array(onp.random.rand(1425) * lambda_init_multiplier)

    opt = optimizers.adam(learning_rate)
    opt_state = opt.init_fn(jnp.concatenate((params0, lambdas0)))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def take_step(step, opt_state):
        params, lambdas = jnp.split(opt.params_fn(opt_state), [100,])
        value = lagrangian(params, lambdas, not_grad=True)

        print("Lambdas >= 0:", all(lambdas >= 0), "\tMean:", lambdas.mean())
        coords = onp.array(param2coords(params))
        ax.scatter(coords[:, 0], coords[:, 1])
        fig.canvas.draw()
        plt.pause(1e-6)
        ax.clear()
        save_params(params, 'lagrange_temp.csv')

        grad_params, grad_lambdas = grad_params_lambdas(params, lambdas)
        grad_lambdas = lambda_grad_multiplier * grad_lambdas
        opt_state = opt.update_fn(step, jnp.concatenate((grad_params, grad_lambdas)), opt_state)
        return value, opt_state

    for step in range(num_steps):
        print("Iteration:", step + 1)
        value, opt_state = take_step(step, opt_state)
        print()

    # import pdb; pdb.set_trace()
    final_params, _ = jnp.split(opt.params_fn(opt_state), [100,])
    save_params(final_params, 'jax_lagrange_gd_v0.csv')
    coords = param2coords(final_params)
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.show()

