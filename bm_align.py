from trajectopy_core.alignment.estimation import estimate_alignment
from trajectopy_core.settings.matching import MatchingSettings, MatchingMethod
from trajectopy_core.settings.alignment import AlignmentSettings
from trajectopy_core.settings.processing import ProcessingSettings
from trajectopy_core.trajectory import Trajectory
from trajectopy_core.evaluation.metrics import ate
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from trajectopy_core.plotting.mpl.trajectory_plot import plot_trajectories
def dict_to_table(data: dict):
    """Converts a dictionary to a rich table."""
    table_data = Table(title="ATE Results")
    table_data.add_column("Property")
    table_data.add_column("Value")
    for key, value in data.items():
        table_data.add_row(key, str(value))
    return table_data

if __name__ == "__main__":
    console = Console()
    opti = Trajectory.from_file("../hloc/optitrack_data.traj")
    local = Trajectory.from_file("../hloc/localization_data.traj")
    match = MatchingSettings()
    match.method = MatchingMethod.NEAREST_TEMPORAL
    align = AlignmentSettings()
    align.preprocessing.time_start = 20.0
    align.preprocessing.time_end = 21.0
    align.estimation_settings.all()

    alignment = estimate_alignment(
        traj_from = local,
        traj_to = opti,
        alignment_settings=align,
        matching_settings=match)
    
    traj = local.apply_alignment(alignment, inplace=False)
    traj.name = 'traj'
    plot_trajectories([local, opti])
    plt.show()
    # settings = ProcessingSettings()
    # ate_result = ate(trajectory_gt=opti, trajectory_est=traj, settings=settings)
    # console.print(dict_to_table(ate_result.property_dict))
