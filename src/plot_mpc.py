import json
import utils
import plotting.plotting as plt
import plotting.make_map_images as make_map
import utils.plotting
import numpy as np

# TODO: Use data to determine initial and target poses
# TODO: Make better maps using illustrator
dof = 3


data_file_name = "Brattora05-27-20-43-NMPC-plan-current"
data_folder = "log_data/logs/"
data_file = data_folder + data_file_name + ".json"
save = True
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
theta_actual = data["Config"]["actual theta"]
V_c = data["Config"]["V_c"]
beta_c = data["Config"]["beta"]
# slack = data["slack"]

if abs(V_c) > 0:
    theta_actual.append(V_c * np.cos(beta_c))
    theta_actual.append(V_c * np.sin(beta_c))
    theta_actual.append(0)
    # plt.ravnkloa(show=True)
    # raise Exception("This is dumb")

if save and not use_last_file:
    # fig1, ax1 = plt.plot_vessel_path(
    #     vessel_path, save_file_name=data_file_name)
    plt.subplot(dt, x_pred, u_pred, x_act,
                u_act, save_file_name=data_file_name)
    plt.brattorkaia(path=vessel_path, V_c=V_c, beta_c=beta_c,
                    save_file_name=data_file_name)
    plt.theta_subplot(dt, theta, theta_actual, save_file_name=data_file_name)
    # fig3, axs3 = plt.slack_subplot(dt, slack)

else:
    # fig1, ax1 = plt.plot_vessel_path(
    #     vessel_path)
    # fig3, axs3 = plt.slack_subplot(dt, slack)
    # plt.ravnkloa()
    # make_map.nidelva()
    # make_map.ravnkloa()
    # plt.ravnkloa(path=vessel_path)
    # plt.nidelva()
    if True:
        plt.brattorkaia(path=vessel_path, V_c=V_c, beta_c=beta_c)
        plt.theta_subplot(dt, theta, theta_actual)
        plt.subplot(dt, x_pred, u_pred, x_act, u_act)


plt.show()
