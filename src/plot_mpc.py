import json
import utils
import plotting.plotting as plt
import utils.plotting

data_file_name = "test_mpc_good_sideways_nmpc_only"
data_folder = "log_data/logs/"
data_file = data_folder + data_file_name + ".json"

data = utils.plotting.load_last_file(data_folder)
# data = utils.plotting.load_file_by_name(data_file)

vessel_path = data["Path"]

plt.plot_vessel_path(vessel_path, data_file_name)


dt = data["Config"]["dt"]
x_pred = data["state predictions"]
u_pred = data["control predictions"]
x_act = data["Path"]
u_act = data["u"]

plt.subplot(dt, x_pred, u_pred, x_act, u_act, data_file_name)
