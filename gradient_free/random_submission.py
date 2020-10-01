import os
import numpy as np
import pandas as pd

num_turbines = 50

random = pd.DataFrame({
    'x': np.random.uniform(low=50, high=3950, size=(num_turbines,)),
    'y': np.random.uniform(low=50, high=3950, size=(num_turbines,))
})

random.to_csv('Submissions/random.csv', index=False)
