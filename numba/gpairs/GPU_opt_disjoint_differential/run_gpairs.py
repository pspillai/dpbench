import base_gpairs
import numpy as np
import gaussian_weighted_pair_counts as gwpc
import dpctl, dpctl.tensor as dpt


def ceiling_quotient(n, m):
    return int((n + m - 1) / m)


def count_weighted_pairs_3d_intel_diff(
    n, nbins, d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):
    with dpctl.device_context(base_gpairs.get_device_selector()):
        gwpc.count_weighted_pairs_3d_intel_diff_ker[n, 64](
            n,
            nbins,
            d_x1,
            d_y1,
            d_z1,
            d_w1,
            d_x2,
            d_y2,
            d_z2,
            d_w2,
            d_rbins_squared,
            d_result,
        )
        gwpc.count_weighted_pairs_3d_intel_diff_agg_ker[
            nbins,
        ](d_result, n)


def run_gpairs(
    n, nbins, d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):
    count_weighted_pairs_3d_intel_diff(
        n,
        nbins,
        d_x1,
        d_y1,
        d_z1,
        d_w1,
        d_x2,
        d_y2,
        d_z2,
        d_w2,
        d_rbins_squared,
        d_result,
    )


base_gpairs.run("Gpairs Dppy kernel", run_gpairs)
