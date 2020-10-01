from windrose import WindroseAxes, WindAxes
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


directory = "./Shell_Hackathon Dataset/Wind Data"
for file in os.listdir(directory):
    data = pd.read_csv(os.path.join(directory, file))
    year = file[-8:-4]

    ax = WindAxes.from_ax()
    ax, params = ax.pdf(data['sped'], bins=np.arange(0, 30, 2)[1:])
    ax.set_title("Year: " + year + "  " +str(params))
    plt.savefig("Plots/dist_" + year)
