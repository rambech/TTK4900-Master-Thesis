import json
import utils
import plotting.plotting as plt
import utils.plotting

data_file_name = "Simulator05-07-16-26"
data_folder = "log_data/logs/"
data_file = data_folder + data_file_name + ".json"

data = utils.plotting.load_last_file(data_folder)
# data = utils.plotting.load_file_by_name(data_file)

vessel_path = data["Path"]
dt = data["Config"]["dt"]
x_pred = data["state predictions"]
u_pred = data["control predictions"]
x_act = data["Path"]
u_act = data["u"]
slack = data["slack"]

fig1, ax1 = plt.plot_vessel_path(vessel_path)
fig2, axs2 = plt.subplot(dt, x_pred, u_pred, x_act, u_act)
# fig3, axs3 = plt.slack_subplot(dt, slack)

plt.show()
