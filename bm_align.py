import copy
import os
import logging
import sys
import evo.core.lie_algebra as lie
from evo.core import trajectory
from evo.tools import plot, file_interface, log
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    take_name = input('Insert name of take: ')
    logger = logging.getLogger('evo')
    logging.configure_logging(verbose=True)
    traj_ref = file_interface.read_tum_trajectory_file(f'takes/{take_name}/optitrack_timed.txt')
    traj_est = file_interface.read_tum_trajectory_file(f'takes/{take_name}/localization/localization_data.txt')
    traj_est.transform(lie.se3(np.eye(3), np.array([0, 0, 0])))
    traj_est.scale(0.5)
    logger.info("\nUmeyama alignment with scaling")
    traj_est_aligned_scaled = copy.deepcopy(traj_est)
    traj_est_aligned_scaled.align(traj_ref, correct_scale=True)
    file_interface.write_tum_trajectory_file(f'takes/{take_name}/localization/localization_aligned.txt', traj_est_aligned_scaled, confirm_overwrite=True)
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xyz
    ax = plot.prepare_axis(fig, plot_mode, subplot_arg=221)
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray')
    plot.traj(ax, plot_mode, traj_est, '-', 'blue')
    fig.axes.append(ax)
    plt.title('not aligned')
    ax = plot.prepare_axis(fig, plot_mode, subplot_arg=223)
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray')
    plot.traj(ax, plot_mode, traj_est_aligned_scaled, '-', 'blue')
    fig.axes.append(ax)
    plt.title('$\mathrm{Sim}(3)$ alignment')
    fig.tight_layout()
    plt.show()
