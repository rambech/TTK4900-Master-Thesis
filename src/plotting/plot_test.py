import matplotlib.pyplot as plt
import json

from matplotlib import patches
from matplotlib.transforms import Affine2D
import numpy as np

file_name = "test/forward.json"

with open(file_name, 'r') as file:
    dict = json.load(file)


def crunch_numbers(dict):
    num_success = 0
    num_crashes = 0
    num_timeouts = 0
    total_reward = 0
    timesteps_used = 0
    for idx in range(100):
        if dict[f"{idx}"]["Termination state"] == "Success":
            num_success += 1
        elif dict[f"{idx}"]["Termination state"] == "Crashed":
            num_crashes += 1
        elif dict[f"{idx}"]["Termination state"] == "Timeout":
            num_timeouts += 1

        total_reward += dict[f"{idx}"]["Total reward"]
        timesteps_used += len(dict[f"{idx}"]["Psi"])

    avg_total_reward = total_reward/100
    avg_timesteps_used = timesteps_used/100

    print(f"Observation: \n \
                    reward:         {avg_total_reward} \n \
                    timesteps:      {avg_timesteps_used} \n \
                    num_success:    {num_success} \n \
                    num_crashes:    {num_crashes} \n \
                    num_timeouts:   {num_timeouts} \n")


def plot_test(dict):
    # Latex settings for plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    pos = (0, 0)
    psi = np.pi/8
    target_pos = (0, 15-0.75-0.5)
    target_psi = np.pi/2
    file_name = "sideways-docking-path"

    def otter(pos, psi, alpha):
        sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                    [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                    [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                    [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
        rotation = Affine2D().rotate(psi)
        translation = Affine2D().translate(pos[0], pos[1])
        boat = patches.Polygon(
            sequence, closed=True, edgecolor='#90552a', facecolor='#f4ac67', linewidth=0.5, alpha=alpha)
        transform = rotation + translation + ax.transData
        boat.set_transform(transform)
        return boat

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
    # asv = ax.add_patch(otter(pos, psi))

    for idx in range(100):
        if dict[f"{idx}"]["Termination state"] == "Success":
            north = dict[f"{idx}"]["North pos"]
            east = dict[f"{idx}"]["East pos"]
            psi = dict[f"{idx}"]["Psi"]
            p, = ax.plot(east, north, color="#2e7578")
            for j in range(len(psi)):
                if j % 10 == 0:
                    pos = (east[j], north[j])
                    ax.add_patch(otter(pos, psi[j], alpha=0.3))

    ax.legend([q, rest, b, otter((0, 0), np.pi/2, 1), p], ["Permitted area",
                                                           "Restricted area", r'$\mathbb{S}_h$', "ASV", "Path"], loc="upper left")

    ax.set(xlim=(-20, 20), ylim=(-15, 15),
           xlabel='E', ylabel='N')

    if False:
        ax.text(13, 5, r'$\mathbb{S}$', fontsize=12)

    plt.savefig(f'figures/{file_name}.pdf', bbox_inches='tight', dpi=400)
    plt.show()


# crunch_numbers(dict)
plot_test(dict)
