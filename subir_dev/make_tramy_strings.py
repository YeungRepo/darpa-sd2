#! /usr/env/bin python

import numpy as np;
all_times = np.arange(0,96.25,0.25);

all_times = [repr(x) for x in all_times]
print(','.join(all_times)) + ' hr'
