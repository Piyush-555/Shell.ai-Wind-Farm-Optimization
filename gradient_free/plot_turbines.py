import numpy as np
import matplotlib.pyplot as plt

import Farm_Evaluator_Vec as evaluator

path = "Submissions/fixed_2000.csv"
turb_coords = evaluator.getTurbLoc(path)
x, y = turb_coords[:, 0], turb_coords[:, 1]
plt.scatter(x, y)
plt.show()
