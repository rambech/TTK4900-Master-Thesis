import json
import utils
import plotting.plotting as plt
import plotting.make_map_images as make_map
import utils.plotting

# TODO: Use data to determine initial and target poses
# TODO: Make better maps using illustrator


data_file_name = "Simulator05-22-14-41-RL-NMPC-without-SYSID"
data_folder = "log_data/logs/"
data_file = data_folder + data_file_name + ".json"
save = False
use_last_file = True

if use_last_file:
    data = utils.plotting.load_last_file(data_folder)
else:
    data = utils.plotting.load_file_by_name(data_file)

vessel_path = data["Path"]
dt = data["Config"]["dt"]
x_pred = data["state predictions"]
u_pred = data["control predictions"]
x_act = data["Path"]
u_act = data["u"]
theta = data["parameters"]
# slack = data["slack"]

# plt.ravnkloa(show=True)
# raise Exception("This is dumb")

if save and not use_last_file:
    # fig1, ax1 = plt.plot_vessel_path(
    #     vessel_path, save_file_name=data_file_name)
    fig2, axs2 = plt.subplot(dt, x_pred, u_pred, x_act,
                             u_act, save_file_name=data_file_name)
    plt.brattorkaia(path=vessel_path, save_file_name=data_file_name)
    plt.theta_subplot(dt, theta, save_file_name=data_file_name)
    # fig3, axs3 = plt.slack_subplot(dt, slack)

else:
    # fig1, ax1 = plt.plot_vessel_path(
    #     vessel_path)
    fig2, axs2 = plt.subplot(dt, x_pred, u_pred, x_act,
                             u_act)
    # fig3, axs3 = plt.slack_subplot(dt, slack)
    # plt.ravnkloa()
    # make_map.nidelva()
    # plt.ravnkloa(path=vessel_path)
    plt.brattorkaia(path=vessel_path)
    plt.theta_subplot(dt, theta)


plt.show()
