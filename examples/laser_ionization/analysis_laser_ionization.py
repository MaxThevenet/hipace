#! /usr/bin/env python3

#
# This file is part of HiPACE++.
#
# Authors:

import argparse
import numpy as np
import math
from openpmd_viewer import OpenPMDTimeSeries
import statistics

parser = argparse.ArgumentParser(description='Compare with WarpX the fraction of ionization for a specific value of a0 with a linear polarized laser')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    help='Path to the directory containing output files')
args = parser.parse_args()

ts = OpenPMDTimeSeries(args.output_dir)

lambda0 = 800.e-9
a0 = 0.00885126
nc = 1.75e27
n0 = nc / 10000

me = 9.1093837015e-31
c = 299792458
qe = 1.602176634e-19
C = me * c * c * 2 * math.pi / (lambda0 * qe)

E0 = C * a0

iteration = 0
rho_elec, _ = ts.get_field(field='rho_elec', coord='z', iteration=iteration, plot=False)
rho_elec_mean = np.mean(rho_elec, axis=(1, 2))
rho_average = statistics.mean(rho_elec_mean[0:10])
fraction = rho_average / (-qe) / (n0)

fraction_warpx = 0.41014984 #result from WarpX simulation

relative_diff = np.abs( ( fraction - fraction_warpx ) / fraction_warpx )
tolerance = 0.15
print("percentage error for the fraction of ionization = "+ str(relative_diff *100) + '%')

assert (relative_diff < tolerance), 'Test laser_ionization did not pass'
