import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

case = "urStar-rt100"

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Horizontally stacked subplots')
for perfect_str in ["perfect", "imperfect"]:
    # read the csv file "load_disp.csv" and plot it
    df = pd.read_csv(f"{case}/load-disp-{perfect_str}.csv")
    min_load = df[["minS11"]].to_numpy()
    disp = df[["lambda/lamLin"]].to_numpy()

    freq = 1
    ax1.plot(disp[0::freq], min_load[0::freq], "o-", linewidth=2, label=perfect_str)
    ax1.plot( [disp[0], disp[-1]], [min_load[0], min_load[1]*disp[-1]/disp[1]], "o--", color='tab:blue', linewidth=2)

    # plot the 2nd derivatives
    print(f"{min_load.shape=}")
    min_load_arr = np.array(min_load[:,0])
    print(f"{np.diff(min_load_arr)=}")
    deriv2_min_load = np.diff(np.diff(min_load_arr))
    print(f"{deriv2_min_load=}")
    ax2.plot(disp[:-2,0], deriv2_min_load[:], label=perfect_str)

ax1.legend()
ax2.legend()
# plt.plot( [disp[0], disp[-1]], [max_load[0], max_load[1]*disp[-1]/disp[1]], "o--", color='tab:green', linewidth=2)
# plt.plot( [disp[0], disp[-1]], [min_load[0], min_load[1]*disp[-1]/disp[1]], "o--", linewidth=2)
plt.xlabel("lambda")
plt.ylabel("minS11")
plt.show()
