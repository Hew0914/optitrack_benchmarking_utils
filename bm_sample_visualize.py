import plotly.graph_objects as go
import pycolmap
from hloc.hloc.utils import viz_3d
import numpy as np
from pathlib import Path

def get_frame_num_from_time(first_frame, time, base=4):
    return first_frame + round(base * round(time/base), 2)

if __name__ == '__main__':
    take_name = input('Take name: ')
    start_frame = int(input('First frame number: '))
    query_processing_data_dir = Path(take_name) / 'frames'
    dataset_name = 'Arena'
    dataset_path = f'/home/hwillia2/Desktop/spatial/optitrack_benchmarking_utils/maps/{dataset_name}/hloc_data'
    dataset = Path(dataset_path)
    db_reconstruction = dataset / 'scaled_sfm_reconstruction'
    if not db_reconstruction.exists():
        db_reconstruction = dataset / 'sfm_reconstruction'
    reconstruction = pycolmap.Reconstruction(db_reconstruction.__str__())
    # Figure matching both trajectories
    fig1 = viz_3d.init_figure()
    localization = []
    optitrack = []
    with open(f'{take_name}/localization/localization_data.txt') as l, open(f'{take_name}/optitrack_timed.txt') as o:
        while (line := l.readline()) != '':
            row = list(map(float, line.split(' ')))
            localization.append(row)
        while (line := o.readline()) != '':
            row = list(map(float, line.split(' ')))
            optitrack.append(row)
    random_sample = sorted(np.random.choice(len(localization), 30))
    for num in random_sample:
        lrow = localization[num]
        query_image_name = f'photo_{get_frame_num_from_time(start_frame, lrow[0])}.png'
        camera = pycolmap.infer_camera_from_image(query_processing_data_dir / query_image_name)
        pose = pycolmap.Image(tvec=[lrow[1], lrow[2], lrow[3]], qvec=[lrow[4], lrow[5], lrow[6], lrow[7]])
        viz_3d.plot_camera_colmap(fig1, pose, camera, color='rgba(100,255,100,0.5)', name=query_image_name, fill=True)
        orow = optitrack[num]
        query_image_name = f'photo_{get_frame_num_from_time(start_frame, orow[0])}.png'
        camera = pycolmap.infer_camera_from_image(query_processing_data_dir / query_image_name)
        pose = pycolmap.Image(tvec=[orow[1], orow[2], orow[3]], qvec=[orow[4], orow[5], orow[6], orow[7]])
        viz_3d.plot_camera_colmap(fig1, pose, camera, color='rgba(255,100,100,0.5)', name=query_image_name, fill=True)
    fig1.add_trace(go.Scatter3d(
        x = [0,5],
        y = [0,0],
        z = [0,0],
        line=dict(
            color='red',
            width=2
        )
    ))
    fig1.add_trace(go.Scatter3d(
        x = [0,0],
        y = [0,5],
        z = [0,0],
        line=dict(
            color='green',
            width=2
        )
    ))
    fig1.add_trace(go.Scatter3d(
        x = [0,0],
        y = [0,0],
        z = [0,5],
        line=dict(
            color='blue',
            width=2
        )
    ))
    # Figure matching localization to map
    fig2 = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig2, reconstruction, color='rgba(255,0,0,0.5)', points_rgb=True, cameras = False)
    for num in random_sample:
        lrow = localization[num]
        query_image_name = f'photo_{get_frame_num_from_time(start_frame, lrow[0])}.png'
        camera = pycolmap.infer_camera_from_image(query_processing_data_dir / query_image_name)
        pose = pycolmap.Image(tvec=[lrow[1], lrow[2], lrow[3]], qvec=[lrow[4], lrow[5], lrow[6], lrow[7]])
        viz_3d.plot_camera_colmap(fig2, pose, camera, color='rgba(100,255,100,0.5)', name=query_image_name, fill=True)
        orow = optitrack[num]
    fig2.add_trace(go.Scatter3d(
        x = [0,5],
        y = [0,0],
        z = [0,0],
        line=dict(
            color='red',
            width=2
        )
    ))
    fig2.add_trace(go.Scatter3d(
        x = [0,0],
        y = [0,5],
        z = [0,0],
        line=dict(
            color='green',
            width=2
        )
    ))
    fig2.add_trace(go.Scatter3d(
        x = [0,0],
        y = [0,0],
        z = [0,5],
        line=dict(
            color='blue',
            width=2
        )
    ))
    fig1.show()
    fig2.show()