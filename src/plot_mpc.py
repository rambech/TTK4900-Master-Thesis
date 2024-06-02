import json
import utils
import plotting.plotting as plt
import plotting.make_map_images as make_map
import utils.plotting
import numpy as np

# TODO: Use data to determine initial and target poses
# TODO: Make better maps using illustrator
dof = 3


data_file_name = "Brattora05-29-18-27-D-10-B-10-xy-50"
save_file_name = data_file_name
# data_folder = "log_data/logs/"
data_folder = "log_data/logs/nidelva_otter/"
data_file = data_folder + data_file_name + ".json"
save = False
use_last_file = True

if use_last_file:
    data = utils.plotting.load_last_file(data_folder)
else:
    data = utils.plotting.load_file_by_name(data_file)

if not save or (save and use_last_file):
    save_file_name = None

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
cost = data["cost"]
stage_cost = data["stage cost"]
# slack = data["slack"]

theta_actual.append(V_c * np.cos(beta_c))
theta_actual.append(V_c * np.sin(beta_c))
theta_actual.append(0)


if True:
    plt.cost(dt, cost, save_file_name=save_file_name)
    plt.stage_cost(dt, stage_cost, save_file_name=save_file_name)
    # plt.brattorkaia(path=vessel_path, V_c=V_c, beta_c=beta_c,
    #                 save_file_name=save_file_name)
    plt.nidelva(path=vessel_path, V_c=V_c,
                beta_c=beta_c, save_file_name=save_file_name)
    plt.theta_subplot(dt, theta, theta_actual,
                      save_file_name=save_file_name)
    plt.subplot(dt, x_pred, u_pred, x_act, u_act,
                save_file_name=save_file_name)


plt.show()
