import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
import numpy as np
import json

# Latex settings for plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")

data_file = "log_data/logs/Simulator03-18-15-36.json"

with open(data_file, "r") as file:
    dict = json.load(file)

# Plot predictions
for pred in dict["state predictions"]:
    x_pred = pred[0][::40]
    y_pred = pred[1][::40]
    ax.plot(y_pred, x_pred, color="#fbdf7b")

# Print path
x_path, y_path = [], []
for point in dict["Path"]:
    x_path.append(point[0])
    y_path.append(point[1])

ax.plot(y_path, x_path, color="#f4ac67")

background = patches.Rectangle(
    (-20, -20), 40, 40, edgecolor='#97d2d4', facecolor='#97d2d4', linewidth=1)
dock = patches.Rectangle((-20, 20-7.5-0.75), 40, 0.75+5,
                         edgecolor='#808080', facecolor='#e6e6e6', linewidth=1)
quay = patches.Rectangle((-5, 25/2-0.75), 10, 2,
                         edgecolor='#00509e', facecolor='#3e628a', linewidth=1, linestyle="-", alpha=0.3)
restricted0 = patches.Rectangle((-15, 25/2-0.75), 10, 2,
                                edgecolor='#595959', facecolor='#000000', linewidth=1, linestyle="-", alpha=0.3)
restricted1 = patches.Rectangle((5, 25/2-0.75), 10, 2,
                                edgecolor='#595959', facecolor='#000000', linewidth=1, linestyle="-", alpha=0.3)
bounds = patches.Rectangle(
    (-15, -25/2), 30, 25, edgecolor="r", facecolor="none", linewidth=1, linestyle="--")

ax.add_patch(background)
ax.add_patch(dock)
ax.add_patch(restricted0)
rest = ax.add_patch(restricted1)
q = ax.add_patch(quay)
b = ax.add_patch(bounds)


ax.legend([q, rest, b], ["Permitted area",
                         "Restricted area", r'$\mathbb{S}_b$'])

ax.set(xlim=(-20, 20), ylim=(-15, 15),
       xlabel='E', ylabel='N')

if False:
    ax.text(13, 5, r'$\mathbb{S}$', fontsize=12)

plt.show()
