import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# r/t of each cylinder design
df = pd.read_csv("../kdfs.csv", skiprows=1)
print(df)
print(df.columns[1])
nelems = df[["nelems"]].to_numpy()
rt_values = df[[" r/t"]].to_numpy()
nasa_kdfs = df[[" nasaKDF"]].to_numpy()
tacs_kdfs = df[[" tacsKDF"]].to_numpy()

plt.vlines(x=50, ymin=0.0, ymax=1.2, color='tab:grey', linestyles='--', linewidth=2)

nelems_unique = np.unique(nelems)
for i_nelems, c_nelems in enumerate(nelems_unique):
    nelems_mask = np.logical_and(nelems == c_nelems, rt_values > 10)
    if i_nelems == 0:
        plt.plot(rt_values[nelems_mask], nasa_kdfs[nelems_mask], "ko--", label="nasa-kdf")
    plt.plot(rt_values[nelems_mask], tacs_kdfs[nelems_mask], "o-", label=f"tacs-{c_nelems/1000:.0f}k")

plt.xlabel("r/t")
plt.xscale('log')
plt.legend()
plt.ylabel("knockdown factor (KDF)")
plt.show()
