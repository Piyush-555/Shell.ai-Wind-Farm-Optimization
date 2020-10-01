import os
import numpy as np
import pandas as pd

import main


num_turbines = 50

coords = np.load("Submissions/cma_best.npy")
coords = main.param2coords(coords)

random = pd.DataFrame({
    'x': coords[:, 0],
    'y': coords[:, 1]
})

random.to_csv('Submissions/cma_best.csv', index=False)
