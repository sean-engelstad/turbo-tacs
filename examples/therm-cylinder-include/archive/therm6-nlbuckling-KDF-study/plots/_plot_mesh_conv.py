import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ml_buckling as mlb

colors = mlb.four_colors4

# r/t of each cylinder design
rt_values = np.array([1000, 500, 300, 100, 50, 25, 10])
nasa_kdfs = np.array([0.223, 0.322, 0.404, 0.581, 0.678, 0.758, 0.82])
# from 10k element data
tacs_kdfs = {
    "5k" : np.array([0.66, 0.68, 0.705, 0.705, 0.725, 0.79, 0.83]),
    "10k" : np.array([0.61, 0.664, 0.677, 0.691, 0.705, 0.763, 0.82]),
    "20k" : np.array([0.611, 0.672, 0.680, 0.686, 0.700, 0.766, 0.822]),
    # "40k" : np.array([0.61, 0.664, 0.677, 0.691, 0.705, 0.763, 0.82]) * 0.25 # debug
}



for itacs,key in enumerate(tacs_kdfs.keys()):
    nelems = key
    tacs_kdf = tacs_kdfs[key]
    plt.plot(rt_values[::-1], tacs_kdf[::-1], "o-", color=colors[itacs], label=f"tacs-{nelems}")
plt.plot(rt_values[::-1], nasa_kdfs[::-1], "ko--", label="nasa-sp8007")

plt.xlabel("r/t")
plt.xscale('log')
plt.legend()
plt.ylabel("knockdown factor (KDF)")
plt.show()
