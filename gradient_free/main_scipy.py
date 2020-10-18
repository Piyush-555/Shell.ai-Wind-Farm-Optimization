import os
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
import multiprocessing as mp
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import Farm_Evaluator_Vec as evaluator


def get_constraints(bounds=False):
    constraints = []
    for i in range(50):
        for j in range(i+1, 50):
            def constraint(params):
                turb_params = params.reshape((50, 2))
                return np.linalg.norm(turb_params[i] - turb_params[j]) - 0.1
            constraints.append({'type': 'ineq', 'fun': constraint})
    
    if bounds:
        for k in range(100):
            def constraint_lower(params):
                return params[k] - 0.0125
            def constraint_upper(params):
                return 0.9875 - params[k]
            constraints.append({'type': 'ineq', 'fun': constraint_lower})
            constraints.append({'type': 'ineq', 'fun': constraint_upper})
    return constraints


def objective(params):
    turb_coords = param2coords(params)
    turb_rad = 50

    power_curve = evaluator.loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = evaluator.preProcessing(power_curve)

    aeps = []
    wind_dir = "Shell_Hackathon Dataset/Wind Data"
    for file in os.listdir(wind_dir):
        wind_inst_freq =  evaluator.binWindResourceData(os.path.join(wind_dir, file))
        AEP = evaluator.getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, 
                     n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
        aeps.append(AEP)

    return -np.mean(aeps)/500


def coords2param(turb_coords):
    return turb_coords.flatten() / 4000


def param2coords(params):
    return params.reshape((50, 2)) * 4000


def save_params(params):
    coords = param2coords(params)
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1]
    })
    df.to_csv('Submissions/COBYLAv1.csv', index=False)


if __name__ == "__main__":
    num_iter = 1500

    path = "Submissions/cma_540.73.csv"
    turb_coords = evaluator.getTurbLoc(path)
    params0 = coords2param(turb_coords)
    bounds = [(0.0125, 0.9875)] * 100
    cnstr = get_constraints(bounds=True)
    options = {
        'disp': True,
        'rhobeg': 0.001,
        'maxiter': 10000,
        'catol': 0.0
    }
    result = optimize.minimize(objective, params0, method='COBYLA', bounds=bounds, constraints=cnstr, options=options)
    print(result)
    save_params(result.x)
    coords = param2coords(result.x)
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.show()

