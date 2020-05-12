from joblib import Parallel, delayed 
import numpy as np
import time

def best_power_strategy(i, rider):
    ones = np.ones(5)*i
    zeros = np.zeros(5)

    return ones, zeros


realRiderName=['Rider 1', 'Rider 2', 'Rider 3', 'Rider 11', 'Rider 12', 'Rider 13']
powerLoc = []
speedLoc = []
timeLoc = []
previousSpeedLoc = []
res = Parallel(n_jobs=3)(delayed(best_power_strategy)(i, rider) for i, rider in enumerate(realRiderName))
powerLoc=[item[0] for item in res]
speedLoc=[item[1] for item in res]
timeLoc=[item[2] for item in res]
previousSpeedLoc=[item[3] for item in res]

print powerLoc
print speedLoc
print timeLoc
print previousSpeedLoc