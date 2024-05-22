import matplotlib.pyplot as plt
import numpy as np
import json

# Latex settings for plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')

reward_file_name = "sideways_reward_plot"
length_file_name = "sideways_length_plot"

# rewards = ["log_data/reward/PPO-docking-71-a_PPO_1.json", "log_data/reward/PPO-docking-71-b_PPO_1.json",
#            "log_data/reward/PPO-docking-71-c_PPO_1.json", "log_data/reward/TD3-docking-0_TD3_1.json",
#            "log_data/reward/TD3-docking-0-b_TD3_1.json", "log_data/reward/TD3-docking-0-c_TD3_1.json"]

# lengths = ["log_data/length/PPO-docking-71-a_PPO_1.json", "log_data/length/PPO-docking-71-b_PPO_1.json",
#            "log_data/length/PPO-docking-71-c_PPO_1.json", "log_data/length/TD3-docking-0_TD3_1.json",
#            "log_data/length/TD3-docking-0-b_TD3_1.json", "log_data/length/TD3-docking-0-c_TD3_1.json"]

rewards = ["log_data/reward/PPO-sideways-1-a_PPO_1.json", "log_data/reward/PPO-sideways-1-b_PPO_1.json",
           "log_data/reward/PPO-sideways-1-c_PPO_1.json"]

lengths = ["log_data/length/PPO-sideways-1-a_PPO_1.json",
           "log_data/length/PPO-sideways-1-b_PPO_1.json", "log_data/length/PPO-sideways-1-c_PPO_1.json"]


# , "#90552a", "#f4ac67", "#fdd9b5"]
colors = ["#282d37", "#2e7578", "#97d2d4"]
labels = ["PPO 4", "PPO 5", "PPO 6"]  # , "TD3 1", "TD3 2", "TD3 3"]

fig0, ax0 = plt.subplots(figsize=(7, 7))

for log, color, label in zip(rewards, colors, labels):
    with open(log, 'r') as file:
        dict = json.load(file)

    data = np.array(dict)
    reward = data[:, -1]
    step = data[:, 1]

    ax0.plot(step, reward, color=color, label=label)

ax0.set(xlim=(0, 12000000), ylim=(-10000, 26000),
        xlabel='Steps', ylabel='Average reward per episode')
ax0.legend()

plt.savefig(f'figures/{reward_file_name}.pdf', bbox_inches='tight', dpi=400)
plt.show()
plt.close()

fig1, ax1 = plt.subplots(figsize=(7, 7))

for log, color, label in zip(lengths, colors, labels):
    with open(log, 'r') as file:
        dict = json.load(file)

    data = np.array(dict)
    reward = data[:, -1]
    step = data[:, 1]

    ax1.plot(step, reward, color=color, label=label)


ax1.set(xlim=(0, 12000000), ylim=(0, 2400),
        xlabel='Steps', ylabel='Average episode length')
ax1.legend()

plt.savefig(f'figures/{length_file_name}.pdf', bbox_inches='tight', dpi=400)
plt.show()
