import json
import utils
import plotting.plotting as plt

data_file = "log_data/logs/Simulator04-10-11-14.json"
data_folder = "log_data/logs/"
# with open(data_file, "r") as file:
#     dict = json.load(file)
data = utils.plotting.load_last_file(data_folder)

# Plot predictions
for pred in data["state predictions"]:
    x_pred = pred[0][::40]
    y_pred = pred[1][::40]

vessel_path = data["Path"]

plt.plot_vessel_path(vessel_path)


dt = data["Config"]["dt"]
x_pred = data["state predictions"]
u_pred = data["control predictions"]
x_act = data["Path"]
u_act = data["u"]

# plt.subplot(dt, x_pred, u_pred, x_act, u_act)
