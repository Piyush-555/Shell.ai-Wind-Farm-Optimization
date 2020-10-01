import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


directory = "./Shell_Hackathon Dataset/Wind Data"
for file in os.listdir(directory):
    data = pd.read_csv(os.path.join(directory, file))
    year = file[-8:-4]
    
    res = sns.kdeplot(data=data['drct'])
    plt.show()
