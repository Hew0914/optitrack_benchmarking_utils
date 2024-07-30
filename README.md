# optitrack_benchmarking_utils

This is a basic library for gathering localization data alongside a ground truth dataset and comparing the two.
A general workflow is as follows:
1. Record gt data (optitrack, april tag) alongside photos with bm_record.py
2. Run bm_localize.py on the created photos to get localization poses
3. Run bm_align.py on the datasets to calculate alignment and error

Alongside these files there are a few utilities useful in visualizing data and setting up to record
 - bm_visualize.py and bm_sample_visualize.py will show a 