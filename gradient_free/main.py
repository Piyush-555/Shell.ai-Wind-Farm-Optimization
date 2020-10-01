import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import cma
import Farm_Evaluator_Vec as evaluator


def get_constraints_penalty(turb_coords, verbose=False, turb_diam=100):
    """
    p_perimeter: Penalty for each perimeter constraint violation.
    p_proximity: Penalty for each proximity constraint violation.

    Returns: Negative penalty value
    """
    bound_clrnc      = 50
    prox_constr_viol = 0
    peri_constr_viol = 0
    prox_penalty = 0
    peri_penalty = 0
    
    # create a shapely polygon object of the wind farm
    farm_peri = [(0, 0), (0, 4000), (4000, 4000), (4000, 0)]
    farm_poly = Polygon(farm_peri)
    
    # checks if for every turbine perimeter constraint is satisfied. 
    # increments peri_constr_viol if violated.
    # add penalty to peri_penalty.
    for turb in turb_coords:
        turb = Point(turb)
        inside_farm   = farm_poly.contains(turb)
        correct_clrnc = farm_poly.boundary.distance(turb) >= bound_clrnc
        if (inside_farm == False or correct_clrnc == False):
            peri_constr_viol += 1
            peri_penalty += 20 - 1/5 * farm_poly.boundary.distance(turb) * (1 if inside_farm else -1)
    
    # checks if for every turbines proximity constraint is satisfied. 
    # increments prox_constr_viol if violated.
    # add penalty to peri_penalty.
    for i,turb1 in enumerate(turb_coords):
        for turb2 in turb_coords[i+1:]:
            if  np.linalg.norm(turb1 - turb2) < 4 * turb_diam:
                prox_constr_viol += 1
                prox_penalty += 20 - 1/40 * np.linalg.norm(turb1 - turb2)
    
    if verbose:
        print("Perimeter Constraints Violated:", peri_constr_viol, "\tProximity Constraints Violated:", prox_constr_viol)

    return -(peri_penalty + prox_penalty)


def evaluate_coords(turb_coords):
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

    mean_aep = np.mean(aeps)
    penalty = get_constraints_penalty(turb_coords, verbose=False)
    fitness = mean_aep + penalty

    return fitness, penalty


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
    df.to_csv('Submissions/cma_best_from_fixed_resarted.csv', index=False)


if __name__ == "__main__":
    num_gen = 1500
    population_size = 40
    # patience = 3
    # patience_length = 10

    path = "Submissions/cma_best_from_fixed.csv"
    turb_coords = evaluator.getTurbLoc(path)

    es = cma.CMAEvolutionStrategy(coords2param(turb_coords), 0.0083)
    # queue = [0,]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_fitness = -float('inf')
    for gen in range(1, num_gen + 1):
        solutions = es.ask(population_size)

        pool = mp.Pool(processes=4)
        returns = pool.starmap(evaluate_coords, [[param2coords(p),] for p in solutions])
        pool.close()

        returns = np.array(returns)
        function_values = -returns[:, 0]  # cma-es minimizes the objective

        best_idx = function_values.argmin()  # argmin to find the most positive fitness value
        best_coords = param2coords(solutions[best_idx])
        best_fitness, _ = returns[best_idx]
        print("Generation:", gen)
        print("Generation Mean -> Mean_fitness: {}\t Penalty: {}".format(*returns.mean(axis=0)))
        print("Generation Best -> Mean_fitness: {}\t Penalty: {}".format(*returns[best_idx]))
        get_constraints_penalty(best_coords, verbose=True)
        print()

        if best_fitness > max_fitness:
            save_params(solutions[best_idx])
            max_fitness = best_fitness
        
        ax.scatter(best_coords[:, 0], best_coords[:, 1])
        fig.canvas.draw()
        plt.pause(1e-6)
        ax.clear()

        es.tell(solutions, function_values)
        es.logger.add()

        # If change in Mean_fitness over last patience_length generations is less than patience,
        # Then reinitialize the ES.
        # queue.append(returns.mean(axis=0)[0])
        # if len(queue) == patience_length + 1:
        #     queue.pop(0)
        #     calc_patience = 0
        #     for i in range(len(queue)-1):
        #         calc_patience += abs(queue[i+1] - queue[i])
        #     if calc_patience < patience:
        #         queue = [0,]
        #         path = "Submissions/cma_best.csv"
        #         turb_coords = evaluator.getTurbLoc(path)
        #         es = cma.CMAEvolutionStrategy(coords2param(turb_coords), 0.02)
