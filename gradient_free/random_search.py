import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import main


def save_coords(coords):
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1]
    })
    df.to_csv('Submissions/random_best.csv', index=False)

iterations = 1
turb_diam = 100
turb_rad = turb_diam/2

power_curve = main.evaluator.loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')
n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = main.evaluator.preProcessing(power_curve)

best_coords = None
maxx = 0.0
for i in tqdm(range(iterations)):

    turb_coords = np.empty((50, 2), dtype=np.float32)
    turb_coords[:, 0] = np.random.uniform(50, 3950, size=(50,))
    turb_coords[:, 1] = np.random.uniform(50, 3950, size=(50,))

    aeps = []
    wind_dir = "Shell_Hackathon Dataset/Wind Data"
    for file in os.listdir(wind_dir):
        wind_inst_freq =  main.evaluator.binWindResourceData(os.path.join(wind_dir, file))
        AEP = main.evaluator.getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, 
                        n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
        aeps.append(AEP)
    mean_aep = np.mean(aeps)
    if mean_aep > 500:
        print("Mean AEP:", mean_aep)
    
    if mean_aep > maxx:
        maxx = mean_aep
        best_coords = turb_coords
        save_coords(best_coords)
