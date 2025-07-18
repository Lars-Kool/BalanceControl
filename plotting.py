import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

with open("values80ml_s.txt") as f:
    lines = f.readlines()

data = np.array([[float(val) for val in line.strip().split(' ')] for line in lines])

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.8))
ax[0].plot(data[:, 0], -(data[:, 1] - data[0, 1]),
        label="Water"
        )
ax[0].plot(data[:, 0], -(data[:, 2] - data[0, 2]),
        label="Ethanol"
        )
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Weight (g)")
ax[0].legend()

ax[1].plot(data[:, 0], data[:, 3],
        label="Water"
        )
ax[1].plot(data[:, 0], data[:, 4],
        label="Ethanol"
        )
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Flowrate (ml/min)")
ax[1].legend()

ax[2].plot(data[:, 0], data[:, 5],
        label="Water"
        )
ax[2].plot(data[:, 0], data[:, 6],
        label="Ethanol"
        )
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Pressure (mbar)")
ax[2].legend()
fig.show()


input()

