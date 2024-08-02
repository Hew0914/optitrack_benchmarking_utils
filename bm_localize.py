import csv
import os
from pathlib import Path
from hloc.hloc import extract_features, extractors, matchers, pairs_from_retrieval, match_features
from hloc.hloc.fast_localize import localize
from hloc.hloc.utils.base_model import dynamic_load
from hloc.hloc.utils.io import list_h5_names
import numpy as np
import copy
import argparse
import evo.core.lie_algebra as lie
from evo.core import trajectory
from evo.tools import plot, file_interface, log
import torch
import logging
import re
from enum import Enum

def get_frame_num_from_path(path):
    frame_name = os.path.basename(path)
    return int(re.sub('[^0-9]', '', frame_name))

class LocalizationClient:
    def __init__(self, take_name, map_name, framerate, local_extractor, global_extractor, matcher):
        self.take_name = take_name
        self.map_name = map_name
        self.framerate = framerate
        self.local_extractor = local_extractor
        self.global_extractor = global_extractor
        self.matcher = matcher

    def run(self):
        print('Starting Localization')
        img_dir = f'takes/{self.take_name}/frames'
        dataset_path = f'maps/{self.map_name}/hloc_data'

        # Load global memory - Common data across all maps
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load local feature extractor model (Superpoint)
        local_feature_conf = extract_features.confs[self.local_extractor]
        Model = dynamic_load(extractors, local_feature_conf['model']['name'])
        local_features_extractor_model = Model(local_feature_conf['model']).eval().to(device)

        # Load global descriptor model (Netvlad)
        global_descriptor_conf = extract_features.confs[self.global_extractor]
        Model = dynamic_load(extractors, global_descriptor_conf['model']['name'])
        global_descriptor_model = Model(global_descriptor_conf['model']).eval().to(device)

        # Load matcher model (SuperGlue)
        match_features_conf = match_features.confs[self.matcher]
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
        frame_offset = get_frame_num_from_path(all_image_paths[0])
        # random_sample = np.random.choice(all_image_paths, 20)
        query_processing_data_dir = Path(img_dir)
        not_localized = []
        num_localized = 0
        os.makedirs(f'takes/{self.take_name}/localization', exist_ok=True)

        with open(f'takes/{self.take_name}/localization/localization_data.txt', 'w', newline='') as f:
            for path in all_image_paths:
                time = (get_frame_num_from_path(path) - frame_offset) / self.framerate
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
        with open(f'takes/{self.take_name}/optitrack_untimed.txt') as untimed, open(f'takes/{self.take_name}/optitrack_timed.txt', 'w', newline='') as timed:
            num_read = 0
            while (l := untimed.readline()) != '' and (num_read < num_localized):
                row = l.split(' ')
                time = (int(row[0]) - frame_offset) / self.framerate
                if int(row[0]) not in not_localized and time >= 0.0:
                    new = [time, row[1], row[2], row[3], row[4], row[5], row[6], row[7]]
                    timed.write(' '.join(map(str, new)))
                    num_read += 1
            untimed.close()
            timed.close()
        with open(f'takes/{self.take_name}/localization/frames_not_localized.txt', 'w') as f:
            f.write(str(not_localized))
            f.close()
        print(f'Localized {num_localized} images.')
        print('Starting alignment')
        logger = logging.getLogger('evo')
        logging.configure_logging(verbose=True)
        traj_ref = file_interface.read_tum_trajectory_file(f'takes/{self.take_name}/optitrack_timed.txt')
        traj_est = file_interface.read_tum_trajectory_file(f'takes/{self.take_name}/localization/localization_data.txt')
        traj_est.transform(lie.se3(np.eye(3), np.array([0, 0, 0])))
        traj_est.scale(0.5)
        logger.info("\nUmeyama alignment with scaling")
        traj_est_aligned_scaled = copy.deepcopy(traj_est)
        traj_est_aligned_scaled.align(traj_ref, correct_scale=True)
        file_interface.write_tum_trajectory_file(f'takes/{take_name}/localization/localization_aligned.txt', traj_est_aligned_scaled, confirm_overwrite=True)


def main():
    parser = argparse.ArgumentParser(description='''\
                                    This applet localizes all frames held in the given recording\'s directory in order along with timestamps.
                                    Additionally, the previously recorded gt data is also trimmed and timestamped to match the successfully localized frames.
                                    The gt and localization data are then aligned.
                                    Please use a map created using the openvps map creation tool.''')
    parser.add_argument('recording', type=str, help='Name of recording')
    parser.add_argument('map', type=str, help='Name of map')
    parser.add_argument('-f', type=float, metavar='FRAMERATE', default=240.0, help='Only for localization. Framerate of gt recording. Default 240Hz')
    parser.add_argument('-l', type=str, metavar='LOCAL EXTRACTOR', default='superpoint_aachen', help='Only for localization. Default Superpoint Aachen')
    parser.add_argument('-g', type=str, metavar='GLOBAL EXTRACTOR', default='netvlad', help='Only for localization. Default Netvlad')
    parser.add_argument('-m', type=str, metavar='MATCHER', default='superglue', help='Only for localization. Default Superglue')
    args = parser.parse_args()
    localizer = LocalizationClient(args.recording, args.map, args.f, args.l, args.g, args.m)
    localizer.run()

if __name__ == '__main__':
    main()