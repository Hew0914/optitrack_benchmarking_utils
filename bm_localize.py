import csv
import os
from pathlib import Path
from hloc.hloc import extract_features, extractors, matchers, pairs_from_retrieval, match_features
from hloc.hloc.fast_localize import localize
from hloc.hloc.utils.base_model import dynamic_load
from hloc.hloc.utils.io import list_h5_names
import numpy as np
import torch
import re
from enum import Enum

# Variables for method of localization
LOCAL_FEATURE_EXTRACTOR = 'superpoint_aachen'
GLOBAL_DESCRIPTOR_EXTRACTOR = 'netvlad'
MATCHER = 'superglue'

FILE_OPTIONS = Enum('FILE_OPTIONS', ['TRAJ', 'CSV'])

# Other variables
TAKE_NAME = input('Take name: ') # Name of folder holding recorded frames
MAP_NAME = 'Arena' # Name of folder holding map information
FRAMERATE = 240.0 # Framerate of camera; keep as float
FILETYPE = 1 # Type of file to save localization data to

def get_frame_num_from_path(path):
    frame_name = os.path.basename(path)
    return int(re.sub('[^0-9]', '', frame_name))

if __name__ == "__main__":
    frame_offset = int(input("Number of first recorded frame: "))
    img_dir = f'{TAKE_NAME}/frames'
    dataset_path = f'maps/{MAP_NAME}/hloc_data'

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
    num_localized = 0
    os.makedirs(f'{TAKE_NAME}/localization', exist_ok=True)

    with open(f'{TAKE_NAME}/localization/localization_data.txt', 'w', newline='') as f:
        for path in all_image_paths:
            time = (get_frame_num_from_path(path) - frame_offset) / FRAMERATE
            try:
                img_name = os.path.basename(path)
                ret, log = localize(
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
                tvec = ret['tvec']
                qvec = ret['qvec']
                out = [time, tvec[0], tvec[1], tvec[2], qvec[0], qvec[1], qvec[2], qvec[3]]
                outstring = ' '.join(map(str, out))
                f.write(f'{outstring}\n')
                num_localized += 1
            except:
                not_localized.append(get_frame_num_from_path(path))
        f.close()
        print(num_localized)
    with open(f'{TAKE_NAME}/optitrack_untimed.txt') as untimed, open(f'{TAKE_NAME}/optitrack_timed.txt', 'w', newline='') as timed:
        num_read = 0
        while (l := untimed.readline()) != '' and (num_read < num_localized):
            row = l.split(' ')
            time = (int(row[0]) - frame_offset) / FRAMERATE
            if int(row[0]) not in not_localized and time >= 0.0:
                new = [time, row[1], row[2], row[3], row[4], row[5], row[6], row[7]]
                timed.write(' '.join(map(str, new)))
                num_read += 1
        untimed.close()
        timed.close()
        print(num_read)
    with open(f'{TAKE_NAME}/localization/frames_not_localized.txt', 'w') as f:
        f.write(str(not_localized))
        f.close()
num_read < num_localized