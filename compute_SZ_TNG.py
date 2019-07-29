import numpy as np
import os
import sys
import eagle as E
import read_eagle as read
import h5py as h5
import copy
import matplotlib.pyplot as plt
from sys import argv, exit, path
from tqdm import tqdm

from illustris_tools import TNGSnapshot

path.insert(0, '/home/arijdav1/Dropbox/phd/Code/EAGLE_xray/')
import eagle_tools

C = eagle_tools.constants()

sim = 'TNG100-1'
snapnum = 99

file_loc = '/hpcdata7/arijdav1/halo_data/TNG_SZ_5r500_'

fname = file_loc+sim+'.hdf5'
filestr = sim+'.hdf5'

if not 'testing' in argv: # don't make new files if using for testing
    if os.path.exists(fname):
        if not 'overwrite' in argv:
            print 'Halo file already exists, enter "overwrite" to override'
            exit()
        else:
            if os.path.exists(file_loc+'OLD_'+filestr):
                print 'Deleting old file backup and backing up existing file...'
                os.remove(file_loc+'OLD_'+filestr)
            os.rename(fname,file_loc+'OLD_'+filestr)

    f = h5.File(fname,'w')


fp = '/hpcdata7/arijdav1/halo_data/illustris_TNG100-1_snap099.hdf5'
infile = h5.File(fp,'r')

groupnumbers = infile['GroupNumber']


N = len(groupnumbers)

ypar = np.zeros(N)

print N,' haloes'

gas = TNGSnapshot(sim = sim, snapnum = snapnum)

snap_gns = gas.groupnumbers
snap_r500s = gas.r500

for g in tqdm(range(N)):

    location = np.where(snap_gns==groupnumbers[g])[0][0]

    gas.select(groupnumbers[g],region_size=5.*gas.r500[location])

    temp = gas.load_temperature()
    mass = gas.load('Masses')

    ypar[g] = np.sum((C.thompson_cgs/(511.*C.ergs_per_keV)) * C.boltzmann_cgs * temp * (mass*C.unit_mass_cgs*0.752*1.17/C.m_p_cgs) / C.Mpc_cgs**2)



print 'Saving...'

f.create_dataset('GroupNumber',data=groupnumbers)
f.create_dataset('ComptonYParam',data=np.log10(ypar))

f.close()







