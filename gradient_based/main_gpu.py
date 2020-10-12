import os
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from cma import CMA
import Farm_Evaluator_Vec as evaluator


def get_constraints_penalty(turb_coords, p_perimeter, p_proximity, verbose=False, turb_diam=100):
    """
    p_perimeter: Penalty for each perimeter constraint violation.
    p_proximity: Penalty for each proximity constraint violation.

    Returns: Negative penalty value
    """
    bound_clrnc      = 50
    prox_constr_viol = 0
    peri_constr_viol = 0
    
    # create a shapely polygon object of the wind farm
    farm_peri = [(0, 0), (0, 4000), (4000, 4000), (4000, 0)]
    farm_poly = Polygon(farm_peri)
    
    # checks if for every turbine perimeter constraint is satisfied. 
    # increments peri_constr_viol if violated.
    for turb in turb_coords:
        turb = Point(turb)
        inside_farm   = farm_poly.contains(turb)
        correct_clrnc = farm_poly.boundary.distance(turb) >= bound_clrnc
        if (inside_farm == False or correct_clrnc == False):
            peri_constr_viol += 1
    
    # checks if for every turbines proximity constraint is satisfied. 
    # increments prox_constr_viol if violated.
    for i,turb1 in enumerate(turb_coords):
        for turb2 in turb_coords[i+1:]:
            if  np.linalg.norm(turb1 - turb2) < 4*turb_diam:
                prox_constr_viol += 1
    
    if verbose:
        print("Perimeter Constraints Violated:", peri_constr_viol, "\tProximity Constraints Violated:", prox_constr_viol)

    return -(p_perimeter * peri_constr_viol + p_proximity * prox_constr_viol)


def evaluate_coords(turb_coords):
    turb_rad = 50
    p_perimeter = 10
    p_proximity = 10

    power_curve = evaluator.loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = evaluator.preProcessing(power_curve)

    aeps = []
    wind_dir = "Shell_Hackathon Dataset/Wind Data"
    for file in os.listdir(wind_dir):
        wind_inst_freq =  evaluator.binWindResourceData(os.path.join(wind_dir, file))
        AEP = evaluator.getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, 
                    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
        aeps.append(AEP)

    mean_aep = np.mean(aeps)
    penalty = get_constraints_penalty(turb_coords, p_perimeter, p_proximity, verbose=False)
    fitness = mean_aep + penalty

    return fitness


def coords2param(turb_coords):
    return turb_coords.flatten() / 4000


def param2coords(params):
    return params.numpy().reshape((-1, 50, 2)) * 4000


def evaluate_params(params):
    coords = param2coords(params)
    pool = mp.Pool(processes=4)
    returns = pool.starmap(evaluate_coords, [[p,] for p in coords])
    pool.close()
    return tf.convert_to_tensor(returns, dtype=tf.float32)


if __name__ == "__main__":
    num_gen = 3000
    population_size = 40

    path = "Submissions/original.csv"
    turb_coords = evaluator.getTurbLoc(path)

    def logging_function(cma, logger):
        fitness = cma.best_fitness()
        solution = cma.best_solution()
        print("Generation:", cma.generation + 1)
        print("Generation Best -> Mean_AEP: {}\t Penalty: {}".format(fitness, get_constraints_penalty(solution.reshape((50, 2)) * 4000, 10, 10)))

    cma = CMA(
        initial_solution=coords2param(turb_coords),
        initial_step_size=0.02,
        fitness_function=evaluate_params,
        population_size=population_size,
        callback_function=logging_function
    )

    cma.search(num_gen)
    # es = cma.CMAEvolutionStrategy(coords2param(turb_coords), 0.02)

    # for gen in range(num_gen):
    #     solutions = es.ask(population_size)

    #     pool = mp.Pool(processes=4)
    #     returns = pool.starmap(evaluate_coords, [[param2coords(p),] for p in solutions])
    #     pool.close()
        
    #     returns = np.array(returns)
    #     function_values = -returns[:, 0]  # cma-es minimizes the objective

    #     best_idx = function_values.argmin()  # argmin to find the most positive fitness value
    #     print("Generation:", gen + 1)
    #     print("Generation Best -> Mean_AEP: {}\t Penalty: {}".format(*returns[best_idx]))
    #     print("Generation Mean -> Mean_AEP: {}\t Penalty: {}\n".format(*returns.mean(axis=0)))

    #     np.save("Submissions/best.npy", solutions[best_idx])

    #     es.tell(solutions, function_values)
    #     es.logger.add()
    #     # es.disp()
