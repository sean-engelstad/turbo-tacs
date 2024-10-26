import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# read the csv file "load_disp.csv" and plot it
perfect_str = "imperfect"
df = pd.read_csv(f"load-disp-{perfect_str}.csv")
min_load = df[["minS11"]].to_numpy()
avg_load = df[["avgS11"]].to_numpy()
max_load = df[["maxS11"]].to_numpy()
disp = df[["lambda/lamLin"]].to_numpy()

freq = 1
plt.plot(disp[0::freq], min_load[0::freq], "o-", color='tab:blue', linewidth=2, label="minS11")
plt.plot(disp[0::freq], avg_load[0::freq], "o-", color='tab:green', linewidth=2, label="avgS11")
# plt.plot(disp[0::freq], max_load[0::freq], "o-", color='tab:orange', linewidth=2, label="maxS11")

plt.plot( [disp[0], disp[-1]], [min_load[0], min_load[1]*disp[-1]/disp[1]], "o--", color='tab:blue', linewidth=2)
plt.plot( [disp[0], disp[-1]], [avg_load[0], avg_load[1]*disp[-1]/disp[1]], "o--", color='tab:green', linewidth=2)

plt.legend()
# plt.plot( [disp[0], disp[-1]], [max_load[0], max_load[1]*disp[-1]/disp[1]], "o--", color='tab:green', linewidth=2)
# plt.plot( [disp[0], disp[-1]], [min_load[0], min_load[1]*disp[-1]/disp[1]], "o--", linewidth=2)
plt.xlabel("lambda")
plt.ylabel("S11")
plt.show()
