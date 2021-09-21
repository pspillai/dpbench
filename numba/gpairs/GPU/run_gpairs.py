import base_gpairs
import numpy as np
import gaussian_weighted_pair_counts as gwpc
import numba_dppy
import dpctl

def run_gpairs(d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result,
            result_tile_1, result_tile_2, queue_global, queue_tile_1, queue_tile_2):

    dpctl.set_global_queue(queue_tile_1)
    # launch in tile 1
    event_1 = gwpc.count_weighted_pairs_3d_intel[d_x1.shape[0]//2, numba_dppy.DEFAULT_LOCAL_SIZE](
        0,
        d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
        d_rbins_squared, result_tile_1)

    dpctl.set_global_queue(queue_tile_2)
    event_2 = gwpc.count_weighted_pairs_3d_intel[d_x1.shape[0]//2, numba_dppy.DEFAULT_LOCAL_SIZE](
        d_x1.shape[0]//2,
        d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
        d_rbins_squared, result_tile_2)

    event_1.wait()
    event_2.wait()

    event_3 = gwpc.merge_results[d_result.shape[0], numba_dppy.DEFAULT_LOCAL_SIZE](result_tile_1, result_tile_2, d_result)

    event_3.wait()

base_gpairs.run("Gpairs Dppy kernel",run_gpairs)
