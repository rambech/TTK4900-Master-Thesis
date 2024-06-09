import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
import matplotlib.ticker as mtick
import utils
from rl.rewards import r_pos_e
import numpy as np
from casadi import Opti

# TODO: Add over all error plots similar to that in "Combining sysid with RL-based MPC"
# TODO: Make prediction error plot

# Latex settings for plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')


class AnyObject:
    pass


class AnotherObject:
    pass


class AThirdObject:
    pass


class OtterHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                    [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                    [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                    [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
        scaled_sequence = []
        for point in sequence:
            x = (point[0] + 0.3)*10
            y = (point[1] + 1)*10
            new_point = [y, x]

            scaled_sequence.append(new_point)

        boat = patches.Polygon(
            scaled_sequence, closed=True, edgecolor='#90552a', facecolor='#f4ac67', linewidth=0.5, alpha=1, transform=handlebox.get_transform())
        handlebox.add_artist(boat)
        return boat


class TargetHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                    [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                    [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                    [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
        scaled_sequence = []
        for point in sequence:
            x = (point[0] + 0.3)*10
            y = (point[1] + 1)*10
            new_point = [y, x]

            scaled_sequence.append(new_point)

        boat = patches.Polygon(
            scaled_sequence, closed=True, edgecolor='#90552a', facecolor='#ff0028', linewidth=0.5, alpha=0.6, transform=handlebox.get_transform(), linestyle="--")
        handlebox.add_artist(boat)
        return boat


class DoubleArrowHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        scale = 1.8
        arrow_length = 10
        head_length = 2
        head_width = 3
        between_length = 1.5
        double_arrow_sequence = [[0, arrow_length/2],
                                 [head_width/2, arrow_length/2-head_length],
                                 [0, arrow_length/2],
                                 [-head_width/2, arrow_length/2-head_length],
                                 [0, arrow_length/2],
                                 [0, arrow_length/2-between_length],
                                 [head_width/2, arrow_length/2 -
                                  head_length-between_length],
                                 [0, arrow_length/2-between_length],
                                 [-head_width/2, arrow_length/2 -
                                  head_length-between_length],
                                 [0, arrow_length/2-between_length],
                                 [0, -arrow_length/2]]

        scaled_sequence = []
        for point in double_arrow_sequence:
            x = (point[0] + 1.5) * scale
            y = (point[1] + 6) * scale
            new_point = [y, x]

            scaled_sequence.append(new_point)
        # rotation = Affine2D().rotate(-np.pi/2)
        # transform = rotation + handlebox.get_transform()

        arrow = patches.Polygon(
            scaled_sequence, closed=False, edgecolor='#000000', facecolor='#000000', linewidth=0.7, alpha=1, transform=handlebox.get_transform())
        handlebox.add_artist(arrow)
        return arrow


def plot3d():
    ax = plt.figure().add_subplot(projection='3d')
    # ax1 = plt.figure().add_subplot(projection='3d')
    x = y = np.arange(-15, 15, 0.1)
    X, Y = np.meshgrid(x, y)

    Z = r_pos_e((X-10.75, Y))
    zeros = np.zeros((Z.shape))

    # Plot the 3D surface 'royalblue'
    ax.plot_surface(X, Y, Z, edgecolor="#2e7578", linewidth=0.5, rstride=8, cstride=8,
                    alpha=0.3)
    # ax.plot_surface(X, Y, zeros, edgecolor='#f4ac67', linewidth=0.2, rstride=8, cstride=8,
    #                 alpha=0.1)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    # ax.contour(X, Y, Z, zdir='z', offset=-3, cmap='coolwarm')
    # ax.contour(X, Y, Z, zdir='x', offset=-20, cmap='coolwarm')
    # ax.contour(X, Y, Z, zdir='y', offset=20, cmap='coolwarm')

    ax.set(xlim=(-16, 16), ylim=(-16, 16), zlim=(-1, 1),
           xlabel='N', ylabel='E', zlabel='R')

    # Plot the 3D surface
    # ax1.plot_surface(X, Y, Z1, edgecolor='seagreen', lw=0.5, rstride=8, cstride=8,
    #                  alpha=0.3)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    # ax1.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    # ax1.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    # ax1.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

    # ax1.set(xlim=(-20, 20), ylim=(-20, 20), zlim=(-3, 3),
    #         xlabel='X', ylabel='Y', zlabel='Z')

    file_name = input("Input file name: ")
    plt.savefig(f'figures/{file_name}.pdf', bbox_inches='tight', dpi=400)

    plt.show()


def plot(dt: float, x_data: np.ndarray, u_data: np.ndarray = None, slack_data: np.ndarray = None):
    t_data = np.arange(start=0, stop=(x_data.shape[1]-1)*dt, step=dt)

    # Position plot
    fig0, ax0 = plt.subplots(figsize=(6, 6))
    ax0.plot(x_data[1, :], x_data[0, :])
    ax0.set(xlabel="East", ylabel="North")

    # Position/time plot x
    fig1, ax1 = plt.subplots(figsize=(6, 6))

    ax1.plot(t_data, x_data[0, :-1])
    ax1.set(xlabel="t", ylabel="North")

    # Position/time plot y
    fig2, ax2 = plt.subplots(figsize=(6, 6))

    ax2.plot(t_data, x_data[1, :-1])
    ax2.set(xlabel="t", ylabel="East")

    # Heading plot psi
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.plot(t_data, x_data[2, :-1])
    ax3.set(xlabel="t", ylabel="$\psi$ heading")

    # U
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    ax4.step(t_data, u_data[0, :],  where="post")
    ax4.step(t_data, u_data[1, :], where="post")
    ax4.set(xlabel="t", ylabel="u control")

    # Print rates
    if len(x_data[:, 0]) > 3:
        fig5, ax5 = plt.subplots(figsize=(6, 6))
        ax5.plot(t_data, x_data[3, :-1])
        ax5.set(xlabel="t", ylabel="$u$ surge")

        fig5, ax5 = plt.subplots(figsize=(6, 6))
        ax5.plot(t_data, x_data[4, :-1])
        ax5.set(xlabel="t", ylabel="$v$ sway")

        fig5, ax5 = plt.subplots(figsize=(6, 6))
        ax5.plot(t_data, x_data[5, :-1])
        ax5.set(xlabel="t", ylabel="$r$ heading rate")

    if slack_data is not None:
        pass

    plt.show()


def subplot(dt: float, x_pred, u_pred, x_act, u_act, show=False, save_file_name=None):
    # TODO: Add target pose
    # Ensure arrays
    x_pred = np.asarray(x_pred)
    u_pred = np.asarray(u_pred)
    x_act = np.asarray(x_act).T
    u_act = np.asarray(u_act).T

    t_data = np.arange(start=0, stop=(
        x_act.shape[1])*dt, step=dt)
    N = x_pred[0].shape[1]-1

    if t_data.shape[0] > x_act.shape[1]:
        t_data = t_data[:-1]

    # layout='constrained'
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(6, 6))

    for i, x in enumerate(x_pred):
        for j in range(x.shape[0]-3):
            if i % 4 == 0:
                t_start = i
                t_end = N+i
                interval = t_data[t_start:t_end]
                axs[j].plot(interval, x[j, :len(interval)],
                            color="#97d2d4", linestyle="--", linewidth=1)

    axs[0].plot(t_data, x_act[0, :], color="#2e7578")
    axs[0].set(ylabel="North")
    axs[0].grid(True)

    axs[1].plot(t_data, x_act[1, :], color="#2e7578")
    axs[1].set(ylabel="East")
    axs[1].grid(True)

    axs[2].plot(t_data, x_act[2, :], color="#2e7578")
    axs[2].set(ylabel="Heading")
    axs[2].grid(True)

    for i, u in enumerate(u_pred):
        if i % 4 == 0:
            t_start = i
            t_end = N+i
            interval = t_data[t_start:t_end]
            axs[3].step(interval, u[0, :len(interval)],
                        color="#97d2d4", where="post", linestyle="--", linewidth=1)
            axs[4].step(interval, u[1, :len(interval)],
                        color="#97d2d4", where="post", linestyle="--", linewidth=1)

        # Burnt orange #f4ac67
        # Light blue #97d2d4

    axs[3].step(t_data, u_act[0, :], color="#2e7578", where="post")
    axs[3].set(ylabel=r"$u_{port}$")
    axs[3].grid(True)

    axs[4].step(t_data, u_act[1, :], color="#2e7578", where="post")
    axs[4].set(ylabel=r"$u_{stb}$")
    axs[4].grid(True)

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_subplots.pdf")
        plt.savefig(
            f'figures/{save_file_name}_subplots.pdf',
            bbox_inches='tight', dpi=400
        )

    if show:
        plt.show()
    # else:
    #     return fig, axs


def cost(costs, save_file_name=None):
    # cost = np.asarray(cost)

    # t_data = np.arange(start=0, stop=(
    #     cost[0].shape[0])*dt, step=dt)

    # if t_data.shape[0] > cost.shape[0]:
    #     t_data = t_data[:-1]

    edgecolors = ["#00509e", "#2e7578", "#d90f0f", "#f8ed62"]
    # labels = ["NMPC", "RL-SYSID, ", "RL-SYSID", "RL-SYSID, B=10"]
    labels = ["NMPC", "RL-SYSID, default",
              "RL-SYSID, initial", "RL-SYSID, initial, B=10"]

    fig, ax = plt.subplots(figsize=(3, 3))
    # ax.plot(t_data, cost, color="#2e7578")
    lines = []
    for cost, color in zip(costs, edgecolors):
        line, = ax.plot(cost, color=color)
        lines.append(line)

    ax.set(ylabel=r"total cost", xlabel=r"steps")
    ax.grid()
    ax.legend(lines, labels)

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_cost.pdf")
        plt.savefig(
            f'figures/{save_file_name}_cost.pdf',
            bbox_inches='tight', dpi=400
        )


def stage_cost(dt: float, stage_cost, save_file_name=None):
    stage_cost = np.asarray(stage_cost)

    t_data = np.arange(start=0, stop=(
        stage_cost.shape[0])*dt, step=dt)

    if t_data.shape[0] > stage_cost.shape[0]:
        t_data = t_data[:-1]

    fig, axs = plt.subplots(figsize=(6, 6))

    axs.plot(t_data, stage_cost, color="#2e7578")
    # TODO: Make sure this is the right notation
    axs.set(ylabel=r"Stage cost $L(x_0,u_0)$")

    # Burnt orange #f4ac67
    # Light blue #97d2d4

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_stage_cost.pdf")
        plt.savefig(
            f'figures/{save_file_name}_stage_cost.pdf',
            bbox_inches='tight', dpi=400
        )


def model_error(parameters, actual, save_file_name=None):
    edgecolors = ["#2e7578", "#d90f0f", "#f8ed62"]
    labels = ["RL", "RL-SYSID", "RL-SYSID, B=10"]

    parameters = parameters[1:]

    fig, ax = plt.subplots(figsize=(3, 3), sharex=True)

    actual = np.asarray(actual)

    lines = []
    for parameter, color in zip(parameters, edgecolors):
        model_error = []
        for param in parameter:
            param = np.asarray(param)
            diff = param[:actual.shape[0]].reshape(actual.shape) - actual
            error = np.linalg.norm(diff, 2)
            model_error.append(np.round(error, 2))

        line, = ax.plot(model_error, color=color)
        lines.append(line)
    ax.set(
        ylabel=r"Model error $\lvert \lvert \theta - \theta_d \rvert \rvert$",
        xlabel=r"steps"
    )
    ax.grid()
    ax.legend(lines, labels)

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_model_error.pdf")
        plt.savefig(
            f'figures/{save_file_name}_model_error.pdf',
            bbox_inches='tight', dpi=400
        )


def slack_subplot(dt: float, slack, show=False, save_file_name=None):
    # Ensure arrays
    slack = np.round(np.asarray(slack), 4)

    t_data = np.arange(start=0, stop=(slack.shape[0])*dt, step=dt)
    N = slack[0].shape[1]-1

    fig, axs = plt.subplots(6, 1, sharex=True)

    axs[0].plot(t_data, slack[:, 0, 1])
    axs[0].set(ylabel=r"$x$")

    axs[1].plot(t_data, slack[:, 1, 1])
    axs[1].set(ylabel=r"$y$")

    axs[2].plot(t_data, slack[:, 2, 1])
    axs[2].set(ylabel=r"$u_{\text{port}}$")

    axs[3].plot(t_data, slack[:, 3, 1])
    axs[3].set(ylabel=r"$u_{\text{stb}}$")

    axs[4].plot(t_data, slack[:, 4, 1])
    axs[4].set(ylabel=r"$\Delta u_{\text{port}}$")

    axs[5].plot(t_data, slack[:, 5, 1])
    axs[5].set(ylabel=r"$\Delta u_{\text{stb}}$")

    for i, s in enumerate(slack):
        for j in range(s.shape[0]):
            t_start = i
            t_end = N+i
            interval = t_data[t_start:t_end]
            num = (3*i*s.shape[0]+j*3)/s.shape[0]
            plot_color = (((10+num)/255, (255-num/2)/255, (255-num)/255))
            # axs[j].plot(interval, s[j, :len(interval)],
            #             color=plot_color, linestyle="--", linewidth=1)
            axs[j].scatter(interval, s[j, :len(interval)],
                           color=plot_color, marker="X", linewidth=0.1)

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_slack_subplots.pdf")
        plt.savefig(
            f'figures/{save_file_name}_slack_subplots.pdf',
            bbox_inches='tight', dpi=400
        )

    if show:
        plt.show()
    else:
        return fig, axs


def theta_subplot(dt: float, theta, actual, show=False, save_file_name=None):
    # Ensure arrays
    theta = np.asarray(theta)
    theta = np.round(theta, 4)
    actual = np.asarray(actual)
    mass = True
    damp = True
    thrust = True
    env = True
    cost = True
    error = True

    # env = False
    # cost = False
    # error = False

    # t_data = np.arange(start=0, stop=(theta.shape[0])*dt, step=dt)

    # print(f"t_data.shape[0]: {t_data.shape[0]}")
    # print(f"theta.shape[1]: {theta.shape[0]}")

    # if t_data.shape[0] > theta.shape[0]:
    #     t_data = t_data[:-1]

    if mass:
        fig1, axs1 = plt.subplots(6, 1, sharex=True, figsize=(6, 6))
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

        for i in range(6):
            # print(f"t_data.shape: {t_data.shape}")
            axs1[i].plot(np.round(theta[:, i], 3), color="#2e7578")
            axs1[i].hlines(actual[i], 0, theta[:, i].shape[0],
                           color="#ff0028", linestyle="--")
            axs1[i].grid()

        axs1[0].set(ylabel=r"$m$")
        axs1[0].set(ylim=(0.4*actual[0], 1.2*actual[0]))
        axs1[1].set(ylabel=r"$I_z$")
        axs1[0].set(ylim=(0.4*actual[0], 1.2*actual[0]))
        axs1[2].set(ylabel=r"$m_{\text{cross}}$")
        axs1[2].set(ylim=(0.8*actual[2], 1.5*actual[2]))
        axs1[3].set(ylabel=r"$X_{\dot{u}}$")
        axs1[3].set(ylim=(1.2*actual[3], 0.2*actual[3]))
        axs1[4].set(ylabel=r"$Y_{\dot{v}}$")
        axs1[4].set(ylim=(1.2*actual[4], 0.8*actual[4]))
        axs1[5].set(ylabel=r"$N_{\dot{r}}$")
        axs1[5].set(xlabel=r"steps")
        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_mass.pdf")
            plt.savefig(
                f'figures/{save_file_name}_mass.pdf',
                bbox_inches='tight', dpi=400
            )

    if damp:
        fig2, axs2 = plt.subplots(4, 1, sharex=True, figsize=(6, 6))

        for i in range(4):
            axs2[i].plot(theta[:, i+6], color="#2e7578")
            axs2[i].hlines(actual[i+6], 0, theta[:, i].shape[0],
                           color="#ff0028", linestyle="--")
            axs2[i].set(ylim=(0.8*actual[i+6], 1.2*actual[i+6]))
            axs2[i].grid()

        axs2[0].set(ylabel=r"$X_{u}$")
        axs2[0].set(ylim=(1.2*actual[6], 0.1*actual[6]))
        axs2[1].set(ylabel=r"$Y_{v}$")
        axs2[1].set(ylim=(1.2*actual[7], 0.8*actual[7]))
        axs2[2].set(ylabel=r"$N_{r}$")
        axs2[2].set(ylim=(1.2*actual[8], 0.8*actual[8]))
        axs2[3].set(ylabel=r"$N_{\lvert r \rvert r}$")
        axs2[3].set(ylim=(1.2*actual[9], 0.8*actual[9]))
        axs2[3].set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_damp.pdf")
            plt.savefig(
                f'figures/{save_file_name}_damp.pdf',
                bbox_inches='tight', dpi=400
            )
    # TODO: Fix goal model values for parameter plot, make the lines dashed and find a suitable color
    if thrust:
        fig3, axs3 = plt.subplots(2, 1, sharex=True, figsize=(6, 3))

        for i in range(2):
            axs3[i].plot(theta[:, i+6+4], color="#2e7578")
            axs3[i].hlines(actual[i+6+4], 0,
                           theta[:, i].shape[0], color="#ff0028", linestyle="--")
            axs3[i].grid()

        axs3[0].set(ylabel=r"$K_{p}$")
        axs3[1].set(ylabel=r"$K_{s}$")
        axs3[1].set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_thrust.pdf")
            plt.savefig(
                f'figures/{save_file_name}_thrust.pdf',
                bbox_inches='tight', dpi=400
            )

    if env:
        fig4, axs4 = plt.subplots(2, 1, figsize=(6, 3), sharex=True)

        for i in range(2):
            axs4[i].plot(theta[:, i+6+4+2], color="#2e7578")
            axs4[i].hlines(actual[i+6+4+2], 0,
                           theta[:, i].shape[0], color="#ff0028", linestyle="--")
            axs4[i].grid()

        axs4[0].set(ylabel=r"$w_1$")
        axs4[1].set(ylabel=r"$w_2$")
        # axs4[2].set(ylabel=r"$w_3$")
        axs4[1].set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_env.pdf")
            plt.savefig(
                f'figures/{save_file_name}_env.pdf',
                bbox_inches='tight', dpi=400
            )

    if cost:
        fig5, axs5 = plt.subplots(4, 1, sharex=True, figsize=(6, 6))

        for i in range(4):
            axs5[i].plot(theta[:, i+6+4+2+3], color="#2e7578")
            axs5[i].grid()

        axs5[0].set(ylabel=r"$\lambda_{\theta}$")
        axs5[1].set(ylabel=r"$V_1$")
        axs5[2].set(ylabel=r"$V_2$")
        axs5[3].set(ylabel=r"$V_3$")
        axs5[3].set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_cost.pdf")
            plt.savefig(
                f'figures/{save_file_name}_cost.pdf',
                bbox_inches='tight', dpi=400
            )

    if error:
        fig6, axs6 = plt.subplots(sharex=True)
        model_error = []
        for param in theta:
            diff = param[:actual.shape[0]].reshape(actual.shape) - actual
            error = np.linalg.norm(diff, 2)
            model_error.append(np.round(error, 2))

        axs6.plot(model_error, color="#2e7578")
        axs6.set(
            ylabel=r"Model error $\lvert \lvert \theta - \theta_d \rvert \rvert$")
        axs6.grid()
        axs6.set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_model_error.pdf")
            plt.savefig(
                f'figures/{save_file_name}_model_error.pdf',
                bbox_inches='tight', dpi=400
            )

    if show:
        plt.show()
    # else:
    #     return fig, axs


def multi_theta_subplot(thetas, actual, show=False, save_file_name=None):
    actual = np.asarray(actual)
    mass = False
    damp = False
    thrust = False
    env = True
    cost = False
    error = False
    big_ass = True

    linewidth = 1

    edgecolors = ["#00509e", "#2e7578", "#d90f0f", "#f8ed62"]
    # labels = ["NMPC", "RL-SYSID, ", "RL-SYSID", "RL-SYSID, B=10"]
    labels = ["NMPC", "RL-SYSID, default",
              "RL-SYSID, initial", "RL-SYSID, initial, B=10"]

    if mass:
        fig1, axs1 = plt.subplots(6, 1, sharex=True, figsize=(6, 6))
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        lines = []
        for theta, color in zip(thetas, edgecolors):
            # Ensure ndarray
            theta = np.asarray(theta)
            for i in range(6):
                # print(f"t_data.shape: {t_data.shape}")
                line, = axs1[i].plot(np.round(theta[:, i], 3), color=color)
                axs1[i].hlines(actual[i], 0, theta[:, i].shape[0],
                               color="#ff0028", linestyle="--")
                axs1[i].grid(True)

            lines.append(line)

        fig1.legend(lines, labels)

        axs1[0].set(ylabel=r"$m$")
        axs1[0].set(ylim=(0.4*actual[0], 1.2*actual[0]))
        axs1[1].set(ylabel=r"$I_z$")
        axs1[0].set(ylim=(0.4*actual[0], 1.2*actual[0]))
        axs1[2].set(ylabel=r"$m_{\text{cross}}$")
        axs1[2].set(ylim=(0.8*actual[2], 1.5*actual[2]))
        axs1[3].set(ylabel=r"$X_{\dot{u}}$")
        axs1[3].set(ylim=(1.2*actual[3], 0.2*actual[3]))
        axs1[4].set(ylabel=r"$Y_{\dot{v}}$")
        axs1[4].set(ylim=(1.2*actual[4], 0.8*actual[4]))
        axs1[5].set(ylabel=r"$N_{\dot{r}}$")
        axs1[5].set(xlabel=r"steps")
        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_mass.pdf")
            plt.savefig(
                f'figures/{save_file_name}_mass.pdf',
                bbox_inches='tight', dpi=400
            )

    if damp:
        fig2, axs2 = plt.subplots(4, 1, sharex=True, figsize=(6, 6))

        lines = []
        for theta, color in zip(thetas, edgecolors):
            # Ensure ndarray
            theta = np.asarray(theta)
            for i in range(4):
                line, = axs2[i].plot(theta[:, i+6], color=color)
                axs2[i].hlines(actual[i+6], 0, theta[:, i].shape[0],
                               color="#ff0028", linestyle="--")
                axs2[i].set(ylim=(0.8*actual[i+6], 1.2*actual[i+6]))
                axs2[i].grid(True)

            lines.append(line)

        fig2.legend(lines, labels)

        axs2[0].set(ylabel=r"$X_{u}$")
        axs2[0].set(ylim=(1.2*actual[6], 0.1*actual[6]))
        axs2[1].set(ylabel=r"$Y_{v}$")
        axs2[1].set(ylim=(1.2*actual[7], 0.8*actual[7]))
        axs2[2].set(ylabel=r"$N_{r}$")
        axs2[2].set(ylim=(1.2*actual[8], 0.8*actual[8]))
        axs2[3].set(ylabel=r"$N_{\lvert r \rvert r}$")
        axs2[3].set(ylim=(1.2*actual[9], 0.8*actual[9]))
        axs2[3].set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_damp.pdf")
            plt.savefig(
                f'figures/{save_file_name}_damp.pdf',
                bbox_inches='tight', dpi=400
            )
    # TODO: Fix goal model values for parameter plot, make the lines dashed and find a suitable color
    if thrust:
        fig3, axs3 = plt.subplots(2, 1, sharex=True, figsize=(6, 3))

        lines = []
        for theta, color in zip(thetas, edgecolors):
            # Ensure ndarray
            theta = np.asarray(theta)
            for i in range(2):
                line, = axs3[i].plot(
                    theta[:, i+6+4], color=color, linewidth=linewidth)
                axs3[i].hlines(actual[i+6+4], 0,
                               theta[:, i].shape[0], color="#ff0028",
                               linestyle="--", linewidth=linewidth)
                axs3[i].grid(True)
            lines.append(line)

        # fig3.legend(lines, labels)
        axs3[0].legend(lines, labels)
        axs3[0].set(ylabel=r"$K_{p}$")
        axs3[1].set(ylabel=r"$K_{s}$")
        axs3[1].set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_thrust.pdf")
            plt.savefig(
                f'figures/{save_file_name}_thrust.pdf',
                bbox_inches='tight', dpi=400
            )

    if env:
        fig4, axs4 = plt.subplots(2, 1, figsize=(6, 3), sharex=True)

        lines = []
        for theta, color in zip(thetas, edgecolors):
            # Ensure ndarray
            theta = np.asarray(theta)
            for i in range(2):
                line, = axs4[i].plot(theta[:, i+6+4+2], color=color)
                axs4[i].hlines(actual[i+6+4+2], 0,
                               theta[:, i].shape[0], color="#ff0028", linestyle="--")
                axs4[i].grid(True)

            lines.append(line)

        # fig4.legend(lines, labels, loc="center left")

        axs4[0].set(ylabel=r"$w_1$")
        axs4[1].set(ylabel=r"$w_2$")
        # axs4[2].set(ylabel=r"$w_3$")
        axs4[1].set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_env.pdf")
            plt.savefig(
                f'figures/{save_file_name}_env.pdf',
                bbox_inches='tight', dpi=400
            )

    if cost:
        fig5, axs5 = plt.subplots(4, 1, sharex=True, figsize=(6, 6))

        lines = []
        for theta, color in zip(thetas[1:], edgecolors[1:]):
            # Ensure ndarray
            theta = np.asarray(theta)
            for i in range(4):
                line, = axs5[i].plot(theta[:, i+6+4+2+3], color=color)
                axs5[i].grid(True)

            lines.append(line)

        axs5[1].legend(lines, labels)

        axs5[0].set(ylabel=r"$\lambda_{\theta}$")
        axs5[1].set(ylabel=r"$V_1$")
        axs5[2].set(ylabel=r"$V_2$")
        axs5[3].set(ylabel=r"$V_3$")
        axs5[3].set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_cost_params.pdf")
            plt.savefig(
                f'figures/{save_file_name}_cost_params.pdf',
                bbox_inches='tight', dpi=400
            )

    if error:
        fig6, axs6 = plt.subplots(sharex=True)
        model_error = []
        for param in theta:
            diff = param[:actual.shape[0]].reshape(actual.shape) - actual
            error = np.linalg.norm(diff, 2)
            model_error.append(np.round(error, 2))

        axs6.plot(model_error, color="#2e7578")
        axs6.set(
            ylabel=r"Model error $\lvert \lvert \theta - \theta_d \rvert \rvert$")
        axs6.grid(True)
        axs6.set(xlabel=r"steps")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_model_error.pdf")
            plt.savefig(
                f'figures/{save_file_name}_model_error.pdf',
                bbox_inches='tight', dpi=400
            )

    if big_ass:
        some_length = 6
        # some_length = int(actual.shape[0]/2)

        if False:  # labels[1] == "RL-SYSID, default":
            fig7, axs7 = plt.subplots(8, 2, sharex=True, figsize=(9, 16))
            axs7[6, 0].set(ylabel=r"$w_1$")
            axs7[6, 1].set(ylabel=r"$w_2$")
            # axs4[14].set(ylabel=r"$w_3$")
        else:
            fig7, axs7 = plt.subplots(6, 2, sharex=True, figsize=(9, 12))

        axs7[0, 0].set(ylabel=r"$m$")
        # axs7[0].set(ylim=(0.4*actual[0], 1.2*actual[0]))
        axs7[0, 1].set(ylabel=r"$I_z$")
        # axs7[1].set(ylim=(0.4*actual[0], 1.2*actual[0]))
        axs7[1, 0].set(ylabel=r"$m_{\text{cross}}$")
        # axs7[2].set(ylim=(0.8*actual[2], 1.5*actual[2]))
        axs7[1, 1].set(ylabel=r"$X_{\dot{u}}$")
        # axs7[3].set(ylim=(1.2*actual[3], 0.2*actual[3]))
        axs7[2, 0].set(ylabel=r"$Y_{\dot{v}}$")
        # axs7[4].set(ylim=(1.2*actual[4], 0.8*actual[4]))
        axs7[2, 1].set(ylabel=r"$N_{\dot{r}}$")
        axs7[3, 0].set(ylabel=r"$X_{u}$")
        # axs7[6].set(ylim=(1.2*actual[6], 0.1*actual[6]))
        axs7[3, 1].set(ylabel=r"$Y_{v}$")
        # axs7[7].set(ylim=(1.2*actual[7], 0.8*actual[7]))
        axs7[4, 0].set(ylabel=r"$N_{r}$")
        # axs7[8].set(ylim=(1.2*actual[8], 0.8*actual[8]))
        axs7[4, 1].set(ylabel=r"$N_{\lvert r \rvert r}$")
        # axs7[9].set(ylim=(1.2*actual[9], 0.8*actual[9]))
        axs7[5, 0].set(ylabel=r"$K_{p}$")
        axs7[5, 1].set(ylabel=r"$K_{s}$")
        # axs7[6, 0].set(ylabel=r"$\lambda_{\theta}$")
        # axs7[6, 1].set(ylabel=r"$V_1$")
        # axs7[7, 0].set(ylabel=r"$V_2$")
        # axs7[7, 1].set(ylabel=r"$V_3$")

        lines = []
        for theta, color in zip(thetas, edgecolors):
            # Ensure ndarray
            count = 0
            theta = np.asarray(theta)
            for i in range(some_length):
                for j in range(2):
                    axs7[i, j].grid(True)
                    line, = axs7[i, j].plot(theta[:, count], color=color)
                    # axs7[i, j].hlines(actual[count], 0,
                    #                   theta[:, count].shape[0], color="#ff0028", linestyle="--")
                    count += 1

            lines.append(line)

        # for theta, color in zip(thetas[1:], edgecolors[1:]):
        #     # Ensure ndarray
        #     count = actual.shape[0]
        #     theta = np.asarray(theta)
        #     for i in range(int(actual.shape[0]/2), int(actual.shape[0]/2)+2):
        #         for j in range(2):
        #             axs7[i, j].grid(True)
        #             line, = axs7[i, j].plot(theta[:, count], color=color)

        #             count += 1

        # fig7.legend(lines, labels)
        axs7[-1, 0].set(xlabel=r"steps")
        axs7[-1, 1].set(xlabel=r"steps")
        axs7[3, 1].legend(lines, labels)

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_big_ass.pdf")
            plt.savefig(
                f'figures/{save_file_name}_big_ass.pdf',
                bbox_inches='tight', dpi=400
            )

    if show:
        plt.show()


def plot_solution(dt: float, solution: Opti.solve, x: Opti.variable, u: Opti.variable, slack: Opti.variable = None):
    """
    Plot optimizer solution
    """

    x_data = np.round(solution.value(x), 5)
    u_data = np.round(solution.value(u), 5)

    if slack is not None:
        slack_data = np.round(solution.value(slack), 5)
    else:
        slack_data = None

    plot(dt, x_data, u_data, slack_data)


def double_arrow(pos, psi, scale, ax):
    arrow_length = 13*scale
    head_length = 2*scale
    head_width = 3*scale
    between_length = 1.5*scale
    double_arrow_sequence = [[0, arrow_length/2],
                             [head_width/2, arrow_length/2-head_length],
                             [0, arrow_length/2],
                             [-head_width/2, arrow_length/2-head_length],
                             [0, arrow_length/2],
                             [0, arrow_length/2-between_length],
                             [head_width/2, arrow_length/2 -
                                 head_length-between_length],
                             [0, arrow_length/2-between_length],
                             [-head_width/2, arrow_length/2 -
                                 head_length-between_length],
                             [0, arrow_length/2-between_length],
                             [0, -arrow_length/2]]
    rotation = Affine2D().rotate(-psi)
    translation = Affine2D().translate(pos[0], pos[1])
    arrow = patches.Polygon(
        double_arrow_sequence, closed=False, edgecolor='#000000', facecolor='#000000', linewidth=1)
    transform = rotation + translation + ax.transData
    arrow.set_transform(transform)
    return arrow


def otter(pos, psi, alpha, ax, edgecolor='#90552a', facecolor='#f4ac67'):
    sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
    rotation = Affine2D().rotate(-psi)
    translation = Affine2D().translate(pos[0], pos[1])
    boat = patches.Polygon(
        sequence, closed=True, edgecolor=edgecolor, facecolor=facecolor, linewidth=0.5, alpha=alpha)
    transform = rotation + translation + ax.transData
    boat.set_transform(transform)
    return boat


def safety_bounds(pos, psi, ax):
    buffer = 0.2
    # buffer = 0.0
    sequence = [[buffer+0.5, buffer+1], [buffer+0.5, -buffer-1],
                [-buffer-0.5, -buffer-1], [-buffer-0.5, buffer+1]]
    rotation = Affine2D().rotate(-psi)
    translation = Affine2D().translate(pos[0], pos[1])
    bound = patches.Polygon(
        sequence, closed=True, edgecolor=(62/255, 98/255, 138/255, 1), facecolor=(151/255, 210/255, 212/255, 0), linewidth=0.5, linestyle="--"
    )
    transform = rotation + translation + ax.transData
    bound.set_transform(transform)
    return bound


def target_pose(pos, psi, alpha, ax):
    sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
    rotation = Affine2D().rotate(-psi)
    translation = Affine2D().translate(pos[0], pos[1])
    boat = patches.Polygon(
        sequence, closed=True, edgecolor='#90552a', facecolor='#ff0028', linewidth=0.5, alpha=alpha, linestyle="--")
    transform = rotation + translation + ax.transData
    boat.set_transform(transform)
    return boat


def plot_vessel_path(path, show=False, save_file_name=None):
    """
    Function for plotting vessel path on the simple map

    """
    # Latex settings for plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    pos = (0, 0)
    # psi = np.pi/8
    target_pos = (0, 15-0.75-0.5)
    target_psi = np.pi/2

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

    path = np.asarray(path)
    p, = ax.plot(path[:, 1], path[:, 0], color="#2e7578")

    # north, east, psi = path[-1, :]

    for north, east, psi in path:
        pos = (east, north)
        ax.add_patch(otter(pos, psi, alpha=0.3, ax=ax))

    ax.add_patch(safety_bounds(pos, psi, ax=ax))

    ax.legend([q, rest, b, otter((0, 0), np.pi/2, 1, ax), p], ["Permitted area",
                                                               "Restricted area", r'$\mathbb{S}_h$', "ASV", "Path"], loc="upper left")

    ax.set(xlim=(-20, 20), ylim=(-15, 15),
           xlabel='E', ylabel='N')

    if False:
        ax.text(13, 5, r'$\mathbb{S}$', fontsize=12)

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_vessel_path.pdf")
        plt.savefig(
            f'figures/{save_file_name}_vessel_path.pdf',
            bbox_inches='tight', dpi=400
        )

    if show:
        plt.show()
    else:
        return fig, ax


def show():
    plt.show()


def brattorkaia(paths=None, V_c=0, beta_c=0, show=False, save_file_name=None):
    """
    Map plot of the water within Brattørkaia, Trondheim, Norway

    Map dimensions: (218.98098581121982, 171.260755066673)


    """

    initial = False

    edgecolors = ["#00509e", "#2e7578", "#d90f0f", "#f8ed62"]
    facecolors = ["#6096d0", "#97d2d4", "#fc4444", "#fff9ae"]
    # labels = ["NMPC", "RL", "RL-SYSID", "RL-SYSID, B=10"]
    labels = ["NMPC", "RL-SYSID, default",
              "RL-SYSID, initial", "RL-SYSID, initial, B=10"]

    fig, ax = plt.subplots(figsize=(1.3*6, 6))

    image_file = "plotting/assets/brattora.png"
    image = plt.imread(image_file)
    dimensions = (218.98098581121982, 171.260755066673)
    extent = (
        -dimensions[0]/4, dimensions[0]/4,
        -dimensions[1]/4, dimensions[1]/4
    )
    ax.imshow(image, extent=extent)

    harbour_sequence = [[-42.5, 15],
                        [-12.5, 40],
                        [30, -7.5],
                        [25.5, -16.5],
                        [15, -26],
                        [-2, -30]]
    harbour_bounds = patches.Polygon(
        harbour_sequence, closed=True, edgecolor="r", facecolor="none", linewidth=1, linestyle="--"
    )

    ax.add_patch(harbour_bounds)
    # if view == "inital":
    ax.add_patch(otter((-20.00666667, 23.240456),
                 utils.D2R(137.37324840062468), 1, ax=ax))
    ax.add_patch(target_pose((19.44486, -20.36019),
                 utils.D2R(137.37324840062468), 0.6, ax=ax))

    lines = []
    if paths is not None:
        for path, edge, face, label in zip(paths, edgecolors, facecolors, labels):
            line, = plot_path(path, ax, 5, edge, face, label)
            lines.append(line)

    if abs(V_c) > 0:
        ax.add_patch(double_arrow((-40, -20), beta_c, 0.7, ax))

        if initial:
            ax.legend([harbour_bounds, AnyObject(), AnotherObject(), AThirdObject()],
                      [r'$\mathbb{S}_b$', "ASV",
                          "Target pose", "Ocean Current"],
                      handler_map={AnyObject: OtterHandler(),
                                   AnotherObject: TargetHandler(),
                                   AThirdObject: DoubleArrowHandler()},
                      bbox_to_anchor=(0.992, 0.992))
        else:
            ax.legend(lines, labels, bbox_to_anchor=(0.992, 0.992))

    else:
        if initial:
            ax.legend([harbour_bounds, AnyObject(), AnotherObject(), AThirdObject()],
                      [r'$\mathbb{S}_b$', "ASV",
                          "Target pose", "Ocean Current"],
                      handler_map={AnyObject: OtterHandler(),
                                   AnotherObject: TargetHandler(),
                                   AThirdObject: DoubleArrowHandler()},
                      bbox_to_anchor=(0.992, 0.992))
            ax.add_patch(double_arrow(
                (-40, -20), 0, 0.7, ax))
            ax.add_patch(otter((-20.00666667, 23.240456),
                               utils.D2R(137.37324840062468), 1, ax=ax))
        else:
            ax.legend(lines, labels, bbox_to_anchor=(0.992, 0.992))

    ax.set(xlabel='E', ylabel='N')

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_brattorkaia.pdf")
        plt.savefig(
            f'figures/{save_file_name}_brattorkaia.pdf',
            bbox_inches='tight', dpi=400
        )

    if show:
        plt.show()
    else:
        return fig, ax


def ravnkloa(paths=None, V_c=0, beta_c=0, show=False, save_file_name=None):
    """
    Map plot of the channel by Ravnkloa, Trondheim, Norway

    Map dimensions (656.9629829983534, 513.7822651994743)

    """

    initial = True

    edgecolors = ["#00509e", "#2e7578", "#d90f0f", "#f8ed62"]
    facecolors = ["#6096d0", "#97d2d4", "#fc4444", "#fff9ae"]
    # labels = ["NMPC", "RL", "RL-SYSID", "RL-SYSID, B=10"]
    labels = ["NMPC", "RL-SYSID, default",
              "RL-SYSID, initial", "RL-SYSID, initial, B=10"]

    fig, ax = plt.subplots(figsize=(1.3*6, 6))

    # image_file = "plotting/assets/ravnkloa.png"
    image_file = "plotting/assets/ravnkloa_close_up.png"
    image = plt.imread(image_file)
    # dimensions = (656.9629829983534, 513.7822651994743)
    dimensions = (218.98681992249377, 171.26075506640066)
    extent = (
        -dimensions[0]/4, dimensions[0]/4,
        -dimensions[1]/4, dimensions[1]/4
    )

    ax.imshow(image, extent=extent)

    # harbour_sequence = [[110, 26],
    #                     [0, 1],
    #                     [115, 80]]

    harbour_sequence = [[47, -10],
                        [30, -14],
                        [-45, -20],
                        [40, 37]]
    harbour_bounds = patches.Polygon(
        harbour_sequence, closed=True, edgecolor="r", facecolor="none", linewidth=1, linestyle="--"
    )

    ax.add_patch(harbour_bounds)
    ax.add_patch(target_pose((36.5, -11),
                             utils.D2R(165), 0.6, ax=ax))

    lines = []
    if paths is not None:
        for path, edge, face, label in zip(paths, edgecolors, facecolors, labels):
            line, = plot_path(path, ax, 5, edge, face, label)
            lines.append(line)

    if abs(V_c) > 0:
        ax.add_patch(double_arrow((-40, -20), beta_c, 0.7, ax))

        if initial:
            ax.legend([harbour_bounds, AnyObject(), AnotherObject(), AThirdObject()],
                      [r'$\mathbb{S}_b$', "ASV",
                          "Target pose", "Ocean Current"],
                      handler_map={AnyObject: OtterHandler(),
                                   AnotherObject: TargetHandler(),
                                   AThirdObject: DoubleArrowHandler()},
                      bbox_to_anchor=(0.992, 0.992))
        else:
            ax.legend(lines, labels, bbox_to_anchor=(0.992, 0.992))
            ax.add_patch(double_arrow((-10, 15), utils.D2R(-130), 0.7, ax))
            # ax.add_patch(otter((-30, -15),
            #                    utils.D2R(50), 1, ax=ax))
            ax.add_patch(otter((-15, -7),
                               utils.D2R(50), 1, ax=ax))

    else:
        if initial:
            ax.legend([harbour_bounds, AnyObject(), AnotherObject(), AThirdObject()],
                      [r'$\mathbb{S}_b$', "ASV",
                          "Target pose", "Ocean Current"],
                      handler_map={AnyObject: OtterHandler(),
                                   AnotherObject: TargetHandler(),
                                   AThirdObject: DoubleArrowHandler()},
                      bbox_to_anchor=(0.265, 0.992))
            ax.add_patch(double_arrow((-10, 15), utils.D2R(-130), 0.7, ax))
            ax.add_patch(otter((-15, -7),
                               utils.D2R(50), 1, ax=ax))
        else:
            ax.legend(lines, labels, bbox_to_anchor=(0.186, 0.992))

    # ax.set(xlim=(-5, 120), ylim=(-25, 100), xlabel='E', ylabel='N')
    ax.set(xlabel='E', ylabel='N')

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_ravnkloa.pdf")
        plt.savefig(
            f'figures/{save_file_name}_ravnkloa.pdf',
            bbox_inches='tight', dpi=400
        )
    if show:
        plt.show()


def nidelva(paths=None, V_c=0, beta_c=0, show=False, save_file_name=None):
    """
    Map plot of a narrow part of Nidelva, Trondheim, Norway

    Map dimensions: (175.18429477615376, 222.63898158632085)

    """

    initial = False

    edgecolors = ["#00509e", "#2e7578", "#d90f0f", "#f8ed62"]
    facecolors = ["#6096d0", "#97d2d4", "#fc4444", "#fff9ae"]
    labels = ["NMPC", "RL", "RL-SYSID", "RL-SYSID, B=10"]
    # labels = ["NMPC", "RL-SYSID, default",
    #           "RL-SYSID, initial", "RL-SYSID, initial, B=10"]

    fig, ax = plt.subplots(figsize=(1.3*6, 6))

    image_file = "plotting/assets/nidelva_close.png"
    image = plt.imread(image_file)
    # dimensions = (175.18429477615376, 222.63898158632085)
    dimensions = (218.9803684729781, 171.26075506640066)
    extent = (
        -dimensions[0]/4, dimensions[0]/4,
        -dimensions[1]/4, dimensions[1]/4
    )

    ax.imshow(image, extent=extent)

    harbour_sequence = [[-30, 20],
                        [27.5, 27],
                        [35, 26.2],
                        [35, -30],
                        [-30, -30]]
    harbour_bounds = patches.Polygon(
        harbour_sequence, closed=True, edgecolor="r", facecolor="none", linewidth=1, linestyle="--"
    )

    # if abs(V_c) > 0:
    #     ax.add_patch(double_arrow((-20, 10), beta_c, 0.7, ax))
    #     ax.legend([harbour_bounds, AnyObject(), AnotherObject(), AThirdObject()],
    #               [r'$\mathbb{S}_b$', "ASV", "Target pose", "Ocean Current"],
    #               handler_map={AnyObject: OtterHandler(),
    #                            AnotherObject: TargetHandler(),
    #                            AThirdObject: DoubleArrowHandler()},
    #               bbox_to_anchor=(0.992, 0.992))
    # else:
    #     ax.legend([harbour_bounds, AnyObject(), AnotherObject()],
    #               [r'$\mathbb{S}_b$', "ASV", "Target pose"],
    #               handler_map={AnyObject: OtterHandler(
    #               ), AnotherObject: TargetHandler()},
    #               bbox_to_anchor=(0.992, 0.992))

    # ax.add_patch(double_arrow((-10, 15), utils.D2R(-130), 0.7, ax))
    ax.add_patch(harbour_bounds)
    # ax.scatter(31.25, 26.6, color="red")
    ax.add_patch(target_pose((31.143935018607795, 25.406768959337686),
                             0.10626486289107881, 0.6, ax=ax))

    lines = []
    if paths is not None:
        for path, edge, face, label in zip(paths, edgecolors, facecolors, labels):
            line, = plot_path(path, ax, 2, edge, face, label)
            lines.append(line)

    if abs(V_c) > 0:
        # ax.add_patch(double_arrow((-40, -20), beta_c, 0.7, ax))
        ax.add_patch(double_arrow((-20, 10), utils.D2R(10), 0.7, ax))
        if initial:
            ax.legend([harbour_bounds, AnyObject(), AnotherObject(), AThirdObject()],
                      [r'$\mathbb{S}_b$', "ASV",
                          "Target pose", "Ocean Current"],
                      handler_map={AnyObject: OtterHandler(),
                                   AnotherObject: TargetHandler(),
                                   AThirdObject: DoubleArrowHandler()},
                      bbox_to_anchor=(0.37, 0.992))
            ax.add_patch(otter((-20, -25),
                               0, 1, ax=ax))
        else:
            ax.legend(lines, labels, bbox_to_anchor=(0.37, 0.992))
            ax.add_patch(otter((-20, -25),
                               0, 1, ax=ax))

    else:
        if initial:
            ax.legend([harbour_bounds, AnyObject(), AnotherObject(), AThirdObject()],
                      [r'$\mathbb{S}_b$', "ASV",
                          "Target pose", "Ocean Current"],
                      handler_map={AnyObject: OtterHandler(),
                                   AnotherObject: TargetHandler(),
                                   AThirdObject: DoubleArrowHandler()},
                      bbox_to_anchor=(0.265, 0.992))
            ax.add_patch(double_arrow((-20, 10), utils.D2R(10), 0.7, ax))
            ax.add_patch(otter((-20, -25),
                               0, 1, ax=ax))
        else:
            # ax.legend(lines, labels, bbox_to_anchor=(0.37, 0.992))
            ax.legend(lines, labels, bbox_to_anchor=(0.297, 0.992))

    # ax.set(xlim=(-20, 20), ylim=(-15, 15),
    #        xlabel='E', ylabel='N')
    ax.set(xlabel='E', ylabel='N')

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_nidelva.pdf")
        plt.savefig(
            f'figures/{save_file_name}_nidelva.pdf',
            bbox_inches='tight', dpi=400
        )


def plot_path(path, ax, skip, edgecolor, facecolor, label=None):
    path = np.asarray(path)
    # north, east, psi = path[-1, :]

    for i, (north, east, psi) in enumerate(path):
        if i % skip == 0:
            pos = (east, north)
            ax.add_patch(otter(pos, psi, alpha=0.3, ax=ax,
                         edgecolor=edgecolor, facecolor=facecolor))

    ax.add_patch(otter(pos, psi, alpha=1, ax=ax,
                 edgecolor=edgecolor, facecolor=facecolor))
    # ax.add_patch(safety_bounds(pos, psi, ax=ax))
    if label is not None:
        return ax.plot(path[:, 1], path[:, 0], color=edgecolor, label=label)
    else:
        return ax.plot(path[:, 1], path[:, 0], color=edgecolor)


def plot_huber():
    # TODO: Chose colours and make this real pretty
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    delta = 1
    file_name = "pseudo-Huber-plot"

    if True:
        x = np.arange(-20, 20, 0.01)
        y = delta**2 * (np.sqrt(1 + (x**2)/(delta**2)) - 1)
        y_q = 0.5*x**2

    ax.plot(x, y)    # pseudo-Huber
    ax.plot(x, y_q)  # quadratic
    ax.set(xlim=(-3.5, 3.5), ylim=(0, 3))

    plt.savefig(f'figures/{file_name}.pdf', bbox_inches='tight', dpi=400)
    plt.show()


def prediction_error(prediction_error_list, save_file_name=None):
    state = True
    control = False

    edgecolors = ["#00509e", "#2e7578", "#d90f0f", "#f8ed62"]
    # labels = ["NMPC", "RL-SYSID, ", "RL-SYSID", "RL-SYSID, B=10"]
    labels = ["NMPC", "RL-SYSID, default",
              "RL-SYSID, initial", "RL-SYSID, initial, B=10"]

    # TODO: Make function for plotting both total model error and individual ones

    state_norm_error_list = []
    control_norm_error_list = []
    total_norm_error_list = []
    for prediction_errors in prediction_error_list:
        state_norm_error = []
        control_norm_error = []
        total_norm_error = []
        for prediction_error in prediction_errors:
            # temp = prediction
            # print(f"temp: {temp}")

            error = np.asarray(prediction_error)
            state_norm_e = np.linalg.norm(error[:6], 2)
            state_norm_error.append(state_norm_e)

            control_norm_e = np.linalg.norm(error[6:], 2)
            control_norm_error.append(control_norm_e)

        state_norm_error_list.append(state_norm_error)
        control_norm_error_list.append(control_norm_error)

    if state:
        fig, ax = plt.subplots(figsize=(6, 3))
        lines = []
        for errors, color in zip(state_norm_error_list, edgecolors):
            line, = ax.plot(errors, color=color)
            lines.append(line)

        ax.set(
            ylabel=r"prediction error $\lvert \lvert \pmb{x} - \pmb{x}_s \rvert \rvert$", xlabel=r"steps")
        ax.grid()
        ax.legend(lines, labels)

    if control:
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        lines = []
        for errors, color in zip(control_norm_error_list, edgecolors):
            line, = ax1.plot(errors, color=color)
            lines.append(line)

        ax1.set(
            ylabel=r"prediction error $\lvert \lvert \pmb{u} - \pmb{u}_s \rvert \rvert$", xlabel=r"steps")
        ax1.grid()
        ax1.legend(lines, labels)

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_pred_error.pdf")
        plt.savefig(
            f'figures/{save_file_name}_pred_error.pdf',
            bbox_inches='tight', dpi=400
        )


def step_response(paths, us, target, save_file_name=None):
    # TODO: Add target pose
    # target = np.asarray(target)

    edgecolors = ["#00509e", "#2e7578", "#d90f0f", "#f8ed62"]
    # labels = ["NMPC", "RL-SYSID, ", "RL-SYSID", "RL-SYSID, B=10"]
    labels = ["NMPC", "RL-SYSID, default",
              "RL-SYSID, initial", "RL-SYSID, initial, B=10"]
    linewidth = 0.7

    # layout='constrained'
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(6, 6))

    lines = []
    longest_steps = 1
    for path, u, color in zip(paths, us, edgecolors):
        # Ensure arrays
        path = np.asarray(path).T
        u = np.asarray(u).T

        north_error = abs(path[0, -1] - target[0])
        east_error = abs(path[1, -1] - target[1])
        heading_error = utils.R2D(utils.ssa(path[2, -1] - target[2]))
        print(f"north error: {north_error}")
        print(f"east error: {east_error}")
        print(f"heading error: {heading_error}")

        steps = np.arange(0, u.shape[1])
        if u.shape[1] > longest_steps:
            longest_steps = u.shape[1]

        line, = axs[0].plot(path[0, :], color=color, linewidth=linewidth)
        axs[0].set(ylabel="North")
        axs[0].grid(True)

        axs[1].plot(path[1, :], color=color, linewidth=linewidth)
        axs[1].set(ylabel="East")
        axs[1].grid(True)

        axs[2].plot(path[2, :], color=color, linewidth=linewidth)
        axs[2].set(ylabel="Heading")
        axs[2].grid(True)

        axs[3].step(steps, u[0, :], color=color,
                    where="post", linewidth=linewidth)
        axs[3].set(ylabel=r"$u_{port}$")
        axs[3].grid(True)

        axs[4].step(steps, u[1, :], color=color,
                    where="post", linewidth=linewidth)
        axs[4].set(ylabel=r"$u_{stb}$")
        axs[4].grid(True)

        lines.append(line)

    for i, element in enumerate(target):
        axs[i].hlines(element, 0, longest_steps,
                      color="#ff0028", linestyle="--", linewidth=linewidth)

    axs[0].legend(lines, labels, loc="upper left")
    axs[4].set(xlabel="steps")

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_step_response.pdf")
        plt.savefig(
            f'figures/{save_file_name}_step_response.pdf',
            bbox_inches='tight', dpi=400
        )
