import json
import utils
import plotting.plotting as plt
import plotting.make_map_images as make_map
import utils.plotting
import numpy as np

# scenario = "brattora"
# scenario = "ravnkloa"
scenario = "nidelva"
# vessel = "simple"
vessel = "otter"
# current = "0"
current = "1"
# current = "1_5"
save = True
use_last_file = False
plot_single_file = False
single_file = "log_data/logs/nidelva_otter/Nidelva-RL-SYSID-initial-no-current.json"

if scenario == "brattora":
    if vessel == "simple":
        data_file_name = ["Brattora-NMPC", "Brattora-RL",
                          "Brattora-RL-SYSID", "Brattora-RL-SYSID-B-10"]
    elif vessel == "otter":
        data_file_name = ["Brattora-NMPC-no-current", "Brattora-RL-SYSID-default-no-current",
                          "Brattora-RL-SYSID-initial-no-current", "Brattora-RL-SYSID-initial-B-10-no-current"]
        if current == "1_5":
            data_file_name = ["Brattora-NMPC-1-5-knots", "Brattora-RL-SYSID-default-1-5-knots",
                              "Brattora-RL-SYSID-initial-1-5-knots", "Brattora-RL-SYSID-B-10-initial-1-5-knots"]
        elif current == "1":
            data_file_name = ["Brattora-NMPC-1", "Brattora-RL-SYSID-default-1",
                              "Brattora-RL-SYSID-initial-1", "Brattora-RL-SYSID-initial-B-10-1"]
        else:
            print(f"No test using: {current} knots, plotting without current")

    else:
        print(f"No such vessel: {vessel}")

elif scenario == "ravnkloa":
    if vessel == "simple":
        data_file_name = ["Ravnkloa-NMPC", "Ravnkloa-RL",
                          "Ravnkloa-RL-SYSID", "Ravnkloa-RL-SYSID-B-10"]
    elif vessel == "otter":
        data_file_name = ["Ravnkloa-NMPC-no-current", "Ravnkloa-RL-SYSID-default-no-current",
                          "Ravnkloa-RL-SYSID-initial-no-current", "Ravnkloa-RL-SYSID-initial-B-10-no-current"]
        if current == "1_5":
            data_file_name = ["Ravnkloa-NMPC-1-5-knots", "Ravnkloa-RL-SYSID-default-1-5-knots",
                              "Ravnkloa-RL-SYSID-initial-1-5-knots", "Ravnkloa-RL-SYSID-initial-1-5-knots-B-10"]
        else:
            print(f"No test using: {current} knots, plotting without current")

    else:
        print(f"No such vessel: {vessel}")

elif scenario == "nidelva":
    if vessel == "simple":
        data_file_name = ["Nidelva-NMPC", "Nidelva-RL",
                          "Nidelva-RL-SYSID", "Nidelva-RL-SYSID-B-10"]
    elif vessel == "otter":
        data_file_name = ["Nidelva-NMPC-no-current", "Nidelva-RL-SYSID-default-no-current",
                          "Nidelva-RL-SYSID-initial-no-current", "Nidelva-RL-SYSID-initial-B-10-no-current"]
        if current == "1_5":
            data_file_name = ["Nidelva-NMPC-1-5-knots", "Nidelva-RL-SYSID-default-1-5-knots",
                              "Nidelva-RL-SYSID-initial-1-5-knots", "Nidelva-RL-SYSID-initial-B-10-1-5-knots"]
        elif current == "1":
            data_file_name = ["Nidelva-NMPC-1", "Nidelva-RL-SYSID-default-1",
                              "Nidelva-RL-SYSID-initial-1", "Nidelva-RL-SYSID-initial-B-10-1"]
        else:
            print(f"No test using: {current} knots, plotting without current")

    else:
        print(f"No such vessel: {vessel}")

else:
    print(f"No such scenario: {scenario}")

if vessel == "otter":
    save_file_name = f"{vessel}_{scenario}_{current}"
else:
    save_file_name = f"{vessel}_{scenario}"

data_folder = f"log_data/logs/{scenario}_{vessel}/"

data_file = []
for file in data_file_name:
    data_file.append(data_folder + file + ".json")

if not save or (save and use_last_file):
    save_file_name = None
elif plot_single_file:
    save_file_name = single_file

if not use_last_file and len(data_file) > 1 and not plot_single_file:
    paths = []
    costs = []
    parameters = []
    for file in data_file:
        print(file)
        data_point = utils.plotting.load_file_by_name(file)
        path = data_point["Path"]
        cost = data_point["cost"]
        parameter = data_point["parameters"]
        paths.append(path)
        costs.append(cost)
        parameters.append(parameter)

    actual = data_point["Config"]["actual theta"]

    V_c = data_point["Config"]["V_c"]
    beta_c = data_point["Config"]["beta_c"]
    # plt.model_error(parameters, actual, save_file_name=save_file_name)
    plt.cost(costs, save_file_name=save_file_name)

    if scenario == "brattora":
        plt.brattorkaia(paths, V_c=V_c, beta_c=beta_c,
                        save_file_name=save_file_name)
    elif scenario == "ravnkloa":
        plt.ravnkloa(paths, V_c=V_c, beta_c=beta_c,
                     save_file_name=save_file_name)
    elif scenario == "nidelva":
        plt.nidelva(paths, V_c=V_c, beta_c=beta_c,
                    save_file_name=save_file_name)

else:
    if use_last_file:
        data = utils.plotting.load_last_file("log_data/logs/")
    else:
        data = utils.plotting.load_file_by_name(single_file)

    vessel_path = [data["Path"]]
    dt = data["Config"]["dt"]
    x_pred = data["state predictions"]
    u_pred = data["control predictions"]
    x_act = data["Path"]
    u_act = data["u"]
    theta = data["parameters"]
    theta_actual = data["Config"]["actual theta"]
    V_c = data["Config"]["V_c"]
    beta_c = data["Config"]["beta_c"]
    cost = data["cost"]
    stage_cost = data["stage cost"]
    # slack = data["slack"]

    theta_actual.append(V_c * np.cos(beta_c))
    theta_actual.append(V_c * np.sin(beta_c))
    theta_actual.append(0)

    if True:
        # plt.cost(dt, cost, save_file_name=save_file_name)
        # plt.stage_cost(dt, stage_cost, save_file_name=save_file_name)
        # plt.brattorkaia(paths=vessel_path, V_c=V_c, beta_c=beta_c,
        #                 save_file_name=save_file_name)
        # plt.nidelva(paths=vessel_path, V_c=V_c,
        #             beta_c=beta_c, save_file_name=save_file_name)
        # plt.ravnkloa(paths=vessel_path, V_c=V_c, beta_c=beta_c,
        #              save_file_name=save_file_name)
        plt.theta_subplot(dt, theta, theta_actual,
                          save_file_name=save_file_name)
        plt.subplot(dt, x_pred, u_pred, x_act, u_act,
                    save_file_name=save_file_name)


plt.show()
