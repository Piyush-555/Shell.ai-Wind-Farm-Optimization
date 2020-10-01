from windrose import WindroseAxes, WindAxes
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


directory = "./Shell_Hackathon Dataset/Wind Data"
for file in os.listdir(directory):
    data = pd.read_csv(os.path.join(directory, file))
    year = file[-8:-4]
    
    ax = WindroseAxes.from_ax()
    ax.bar(data['drct'], data['sped'], normed=True, bins=np.arange(0, 30, 2), opening=1.0, edgecolor='white', nsector=36)
    ax.set_legend()
    ax.set_title("Year: " + year)
    plt.xticks(np.pi * np.arange(0, 2, 0.5), list('ENWS'))
    plt.savefig("Plots/windrose_" + year)
