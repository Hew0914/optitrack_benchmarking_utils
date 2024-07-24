import csv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import h5py
from hloc.hloc import extract_features, extractors, matchers, pairs_from_retrieval, match_features
from hloc.hloc.fast_localize import localize
from hloc.hloc.utils.base_model import dynamic_load
from hloc.hloc.utils.io import list_h5_names
import numpy as np
import torch
import re

def get_frame_num_from_path(path):
    frame_name = os.path.basename(path)
    return int(re.sub('[^0-9]', '', frame_name))

if __name__ == "__main__":

    LOCAL_FEATURE_EXTRACTOR = 'superpoint_aachen'
    GLOBAL_DESCRIPTOR_EXTRACTOR = 'netvlad'
    MATCHER = 'superglue'

    # IF ENTERING FRAMES NOT RECORDED USING "record.py", FRAMES MUST BE
    # RENAMED IN "frame_X,png" FORMAT AS TIMESTAMPS ARE ADDED BY FRAME NUMBER

    # Variables
    take_name = 'july_12' # Name of folder holding recorded frames
    dataset_name = 'Arena' # Name of folder holding map information
    framerate = 240.0 # Framerate of camera; keep as float
    time_offset = 0 # Difference between time 0 and frame 0

    img_dir = f'frames/{take_name}'
    dataset_path = f'maps/{dataset_name}/hloc_data'

    # Load global memory - Common data across all maps

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load local feature extractor model (Superpoint)
    local_feature_conf = extract_features.confs[LOCAL_FEATURE_EXTRACTOR]
    Model = dynamic_load(extractors, local_feature_conf['model']['name'])
    local_features_extractor_model = Model(local_feature_conf['model']).eval().to(device)

    # Load global descriptor model (Netvlad)
    global_descriptor_conf = extract_features.confs[GLOBAL_DESCRIPTOR_EXTRACTOR]
    Model = dynamic_load(extractors, global_descriptor_conf['model']['name'])
    global_descriptor_model = Model(global_descriptor_conf['model']).eval().to(device)

    # Load matcher model (SuperGlue)
    match_features_conf = match_features.confs[MATCHER]
    Model = dynamic_load(matchers, match_features_conf['model']['name'])
    matcher_model = Model(match_features_conf['model']).eval().to(device)

    # Load map specific memory

    # Define database paths
    dataset = Path(dataset_path)
    db_local_features_path = (dataset / local_feature_conf['output']).with_suffix('.h5')

    db_reconstruction = dataset / 'scaled_sfm_reconstruction'
    if not db_reconstruction.exists():
        db_reconstruction = dataset / 'sfm_reconstruction'

    # Load global descriptors from the database
    db_global_descriptors_path = (dataset / global_descriptor_conf['output']).with_suffix('.h5')
    db_image_names = np.array(list_h5_names(db_global_descriptors_path))
    db_global_descriptors = pairs_from_retrieval.get_descriptors(db_image_names, db_global_descriptors_path)
    db_global_descriptors = db_global_descriptors.to(device)

    all_image_paths = sorted(Path(img_dir).glob('*.png'), key=get_frame_num_from_path)
    random_sample = np.random.choice(all_image_paths, 20)
    query_processing_data_dir = Path(img_dir)
    not_localized = []

    if not os.path.exists('localizations'):
        os.mkdir('localizations')
    if not os.path.exists(f'localizations/{take_name}'):
        os.mkdir(f'localizations/{take_name}')

    with open(f'localizations/{take_name}/localization_data.csv', 'w', newline='') as f:
        fw = csv.writer(f)
        for path in all_image_paths[:20]:
            time = (get_frame_num_from_path(path) + time_offset) / framerate
            try:
                img_name = os.path.basename(path)
                ret_new, log_new = localize(
                    query_processing_data_dir = query_processing_data_dir, 
                    query_image_name = img_name, 
                    device = device, 
                    local_feature_conf = local_feature_conf, 
                    local_features_extractor_model = local_features_extractor_model, 
                    global_descriptor_conf = global_descriptor_conf, 
                    global_descriptor_model = global_descriptor_model, 
                    db_global_descriptors = db_global_descriptors, 
                    db_image_names = db_image_names,
                    db_local_features_path = db_local_features_path,
                    matcher_model = matcher_model,
                    db_reconstruction = db_reconstruction
                )
                tvec = ret_new['tvec']
                qvec = ret_new['qvec']
                out = [time, tvec[0], tvec[1], tvec[2], qvec[0], qvec[1], qvec[2], qvec[3]]
                fw.writerow(out)
            except:
                not_localized.append(get_frame_num_from_path(path))
        f.close()
    with open(f'localizations/{take_name}/not_localized.txt', 'w') as f:
        f.write(str(not_localized))
        f.close()
    print("Done!")