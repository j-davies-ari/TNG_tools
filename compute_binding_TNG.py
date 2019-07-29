# -*- coding: utf-8 -*-


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

from scipy import integrate

from python_tools import *

from illustris_tools import TNGSnapshot

path.insert(0, '/home/arijdav1/Dropbox/phd/Code/EAGLE_xray/')
import eagle_tools

from multiprocessing import Pool

def compute_potential(params):
    dmmass, dmmasscuml, rcentre = params
    #return 6.6726e-8 * dmmass * dmmasscuml / rcentre
    return -1.* 6.6726e-8 * dmmasscuml / rcentre




save_loc = '/home/arijdav1/Dropbox/phd/figures/paper_plots/scatter_plots/morphology/'


C = eagle_tools.constants()

sim = 'TNG100-1'
snapnum = 99

file_loc = '/hpcdata7/arijdav1/halo_data/TNG_binding_'

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

be_fromDMO = np.zeros(N)
be_fromDMO_integ = np.zeros(N)
be_analytic = np.zeros(N)
be_particle = np.zeros(N)
be_particle_nosf = np.zeros(N)

pe_particle = np.zeros(N)
ke_particle = np.zeros(N)


print N,' haloes'

gas = TNGSnapshot(sim = sim, snapnum = snapnum)


M200 = np.log10(gas.M200 * 1e10) # Log stellar masses
M200_cgs = gas.M200 * C.unit_mass_cgs
r200 = gas.r200 / 1000. # Mpc (EAGLE units)
snap_gns = gas.groupnumbers

massmask = np.where(M200>11.5)[0]

snap_gns = snap_gns[massmask]
r200 = r200[massmask]
M200 = M200[massmask]
M200_cgs = M200_cgs[massmask]



# Match TNG to the DMO run to get a proxy for the intrinsic binding energy

DMO = TNGSnapshot(sim = sim+'-Dark', snapnum = snapnum)

# Don't really need this, it's just to get the length of the SUBFIND lists
ref_Vmax = gas.load_subfind('SubhaloVmax',all_subhalos=True)
# Get the subhalo IDs of all the centrals in our sample
ref_subhaloIDs = np.arange(len(ref_Vmax))[gas.first_subhalo][massmask]

# Load matching files
with h5.File('/hpcdata5/simulations/IllustrisTNG/'+sim+'/postprocessing/subhalo_matching_to_dark.hdf5', 'r') as matchfile:
    
    LHaloTree_matches = np.array(matchfile['Snapshot_%i/SubhaloIndexDark_LHaloTree'%snapnum]) # More rigorous bijective method

DMO_subhaloIDs = LHaloTree_matches[ref_subhaloIDs]

DMO_hostID = DMO.load_subfind('SubhaloGrNr',all_subhalos=True)[DMO_subhaloIDs]

DMO_subhalo_vel = DMO.load_subfind('SubhaloVel',all_subhalos=True)[DMO_subhaloIDs]

DMO_M200 = DMO.load_FOF('Group_M_Crit200',all_groups=True)[DMO_hostID] / DMO.h

DMO_groupnumbers = DMO_hostID + 1

DMO_first_subhalo = DMO.load_FOF('GroupFirstSub',all_groups=True)

# Find any failed matches - either where no match was found (3), or where a central in TNG is not a central in DMO (1)

print 'No match found for ',len(np.where(DMO_subhaloIDs<0.)[0]),' haloes'
print len(np.where(np.in1d(DMO_subhaloIDs,DMO_first_subhalo))[0]),' of ',len(DMO_subhaloIDs),' TNG centrals are also DMO centrals'

# plt.figure(figsize=(8,8))
# plt.scatter(M200,np.log10(DMO_M200*1e10),s=10)
# plt.xlim(11.,15)
# plt.ylim(11.,15)
# plt.show()
# exit()

failures = np.hstack((np.where(DMO_subhaloIDs<0.)[0],np.where(~np.in1d(DMO_subhaloIDs,DMO_first_subhalo))[0]))

snap_gns = np.delete(snap_gns,failures)
M200_cgs = np.delete(M200_cgs,failures)
DMO_M200 = np.delete(DMO_M200,failures)
r200 = np.delete(r200,failures)
DMO_subhalo_vel = np.delete(DMO_subhalo_vel,failures,axis=0)

DMO_groupnumbers = np.delete(DMO_groupnumbers,failures)

for g in tqdm(range(N)):

    # if groupnumbers[g] < 3000:
    #     continue

    location = np.where(snap_gns==groupnumbers[g])[0][0]

    bulk_velocity = DMO_subhalo_vel[location,:] * C.unit_velocity_cgs

    #print 'Analytic BE: ', (0.0486/0.3089) * (3./5.) * C.G_cgs * M200_cgs[location]**2 / (r200[location]*C.unit_length_cgs)

    DMO_gn = DMO_groupnumbers[location]

    DMO.select(DMO_gn,parttype=1,region_size='r200')

    # PE = DMO.load('Potential',cgs=True)
    # vel = DMO.load('Velocities',cgs=True)

    # KE = 0.5 * DMO.masstable[1] * C.unit_mass_cgs * np.einsum('...j,...j->...',vel,vel)

    # be_fromDMO[g] = np.sum(PE * DMO.masstable[1] * C.unit_mass_cgs) + np.sum(KE) # in erg

    boxsize = DMO.boxsize

    dm_coords = DMO.load('Coordinates') - DMO.centre # kpc

    # Wrap the box

    dm_coords+=boxsize/2.
    dm_coords%=boxsize
    dm_coords-=boxsize/2.


    dm_potential = np.zeros(len(dm_coords[:,0]))
    dm_potential_integ = np.zeros(len(dm_coords[:,0]))

    dm_radii = np.sqrt(np.einsum('...j,...j->...',dm_coords,dm_coords))

    # Make the radial bins

    sort_radii = np.argsort(dm_radii)

    sorted_radii = dm_radii[sort_radii] * C.unit_length_cgs/1000.

    sorted_radii[0] = sorted_radii[1]

    dm_masses = np.ones(len(sorted_radii)) * DMO.masstable[1] * C.unit_mass_cgs

    potential_cuml_rev = -1. * C.G_cgs * np.cumsum(dm_masses[::-1]/sorted_radii[::-1])[::-1]

    #potential_cuml_integ = -1. * C.G_cgs * integrate.cumtrapz(dm_masses/sorted_radii**2,sorted_radii,initial=0.)

    potential_cuml_integ = np.zeros(len(sorted_radii))

    dm_mass_cuml = np.cumsum(dm_masses)

    # plt.figure()

    # plt.plot(sorted_radii*1000./C.unit_length_cgs,np.log10(dm_mass_cuml),lw=2)

    # plt.show()



    potential_cuml_integ = C.G_cgs * integrate.cumtrapz(dm_mass_cuml[::-1]/sorted_radii[::-1]**2,sorted_radii[::-1],initial=0.)[::-1]


    '''
    # WORKS 
    print 'N = ',len(sorted_radii)

    for r, radius in tqdm(enumerate(sorted_radii)):

        potential_cuml_integ[r] = -1. * C.G_cgs * np.trapz(dm_mass_cuml[r:]/sorted_radii[r:]**2,sorted_radii[r:])
    '''






    # plt.figure()

    # plt.plot(sorted_radii*1000./C.unit_length_cgs,np.log10(np.absolute(potential_cuml_integ)),lw=2)

    # plt.show()



    #print potential_cuml_integ

    #dm_potential[sort_radii] = potential_cuml_rev * DMO.masstable[1] * C.unit_mass_cgs
    dm_potential_integ[sort_radii] = potential_cuml_integ * DMO.masstable[1] * C.unit_mass_cgs


    vel = DMO.load('Velocities',cgs=True) - bulk_velocity

    dm_KE = 0.5 * DMO.masstable[1] * C.unit_mass_cgs * np.einsum('...j,...j->...',vel,vel)

    be_fromDMO[g] = (0.0486/0.3089) * np.sum(dm_potential + dm_KE)
    be_fromDMO_integ[g] = (0.0486/0.3089) * np.sum(dm_potential_integ + dm_KE)


    #print 'Integrated binding energy (sum): ',be_fromDMO[g]
    #print 'Integrated binding energy (integral): ',be_fromDMO_integ[g]

    #print np.sum(dm_KE), np.sum(dm_potential_integ), (0.0486/0.3089) * np.sum(dm_KE) + np.sum(dm_potential_integ)


    

    # plt.figure()

    # plt.scatter(dm_radii,np.log10(np.absolute(dm_potential_integ)),s=5)

    # #plt.savefig(save_loc+'potential.pdf',bbox_inches='tight',dpi=200)

    # plt.show()



    '''

    # exit()

















    rbins = np.append(sorted_radii[::N_shell],sorted_radii[-1])

    rwidths = (rbins[1:]-rbins[:-1]) * C.unit_length_cgs/1000.

    rcentres = ((rbins[:-1]+rbins[1:])/2.) * C.unit_length_cgs/1000.

    #N_shell = np.histogram(dm_radii,bins=rbins)

    dm_mass_binned = np.float32(N_shell) * np.ones(len(rcentres)) * DMO.masstable[1] * C.unit_mass_cgs


    # dm_mass_binned, be = np.histogram(dm_radii,bins=1000) * DMO.masstable[1] * C.unit_mass_cgs

    # rcentres = get_bincentres(be) * C.unit_length_cgs/1000.

    dm_mass_cuml = np.cumsum(dm_mass_binned)

    dm_mass_reverse_cuml = np.cumsum(dm_mass_binned[::-1])[::-1]

    masstimeswidth_reverse_cuml = np.cumsum((dm_mass_binned[::-1]*rwidths[::-1])/(rcentres[::-1])**2)[::-1]

    potential = -1. * C.G_cgs * DMO.masstable[1] * C.unit_mass_cgs * masstimeswidth_reverse_cuml


    for n in tqdm(range(len(rcentres))):

        dm_potential[(dm_radii>rbins[n])&(dm_radii<rbins[n+1])] = potential[n]



    vel = DMO.load('Velocities',cgs=True)

    dm_KE = 0.5 * DMO.masstable[1] * C.unit_mass_cgs * np.einsum('...j,...j->...',vel,vel)

    dmonly_binding = dm_potential + dm_KE




    print 'Integrated potential: ',(0.0486/0.3089) * np.sum(dmonly_binding)

    # potential = np.zeros(len(rcentres))

    # print len(rcentres)

    # for r, rcentre in tqdm(enumerate(rcentres)):

    #     #to_sum = np.where(rcentres>rcentre)[0]

    #     potential[r] = -1.* 6.6726e-8 * np.sum(dm_mass_binned[r:] * rwidths[r:] / rcentre**2)


    # def particle_potential(params):
    #     dmmassbinned, widths, centres = params

    #     to_sum = np.where(rcentres>rcentre)[0]
    #     potential[r] = -1.* 6.6726e-8 * np.sum(dm_mass_binned[r:] * rwidths[r:] / rcentre**2)

    #     #return 6.6726e-8 * dmmass * dmmasscuml / rcentre
    #     return -1.* 6.6726e-8 * dmmasscuml / rcentre






    plt.figure()

    plt.plot(np.log10(rcentres*1000./C.unit_length_cgs),np.log10(np.absolute(potential)),lw=2)

    #plt.savefig(save_loc+'potential.pdf',bbox_inches='tight',dpi=200)


    plt.show()

    exit()





    # plt.figure()

    # plt.plot(np.log10(rcentres*1000./C.unit_length_cgs),np.log10(dm_mass_cuml*1e10/C.unit_mass_cgs),lw=2)

    # plt.savefig(save_loc+'cuml_mass.pdf',bbox_inches='tight',dpi=200)


    # plt.show()




    params = zip(dm_mass_binned,dm_mass_cuml,rcentres)
    pool = Pool(16)
    shell_potential = pool.map(compute_potential,params)
    pool.close()
    pool.join()

    plt.figure()

    plt.plot(np.log10(rcentres*1000./C.unit_length_cgs),np.log10(np.absolute(shell_potential)),lw=2)

    plt.savefig(save_loc+'potential.pdf',bbox_inches='tight',dpi=200)


    plt.show()

    continue
    '''


    gas.select(groupnumbers[g],region_size='r200')

    be_analytic[g] = (0.0486/0.3089) * (3./5.) * C.G_cgs * M200_cgs[location]**2 / (r200[location]*C.unit_length_cgs)

    PE = gas.load('Potential',cgs=True)
    vel = gas.load('Velocities',cgs=True)
    p_mass = gas.load('Masses',cgs=True)
    SFR = gas.load('StarFormationRate',cgs=True)
    sfmask = np.where(SFR==0.)[0]

    KE = 0.5 * p_mass * np.einsum('...j,...j->...',vel,vel)

    be_particle[g] = np.sum(PE*p_mass) + np.sum(KE) # in erg

    pe_particle[g] = np.sum(PE*p_mass) 

    ke_particle[g] = np.sum(KE)

    PE = PE[sfmask]
    p_mass = p_mass[sfmask]
    vel = vel[sfmask,:]

    KE = 0.5 * p_mass * np.einsum('...j,...j->...',vel,vel)

    be_particle_nosf[g] = np.sum(PE*p_mass) + np.sum(KE) # in erg


print 'Saving...'

f.create_dataset('GroupNumber',data=groupnumbers)
f.create_dataset('be_fromDMO_integ',data=be_fromDMO_integ)
f.create_dataset('be_particle',data=be_particle)
f.create_dataset('be_particle_nosf',data=be_particle_nosf)
f.create_dataset('pe_particle',data=pe_particle)
f.create_dataset('ke_particle',data=ke_particle)
f.create_dataset('be_analytic',data=be_analytic)

f.close()







