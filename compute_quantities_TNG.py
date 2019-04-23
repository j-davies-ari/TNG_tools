# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import eagle as E
import read_eagle as read
import h5py as h5
import copy
import matplotlib.pyplot as plt
from sys import argv
from sys import exit
from tqdm import tqdm

from copy import deepcopy

import astropy.units as u
from astropy.cosmology import Planck13, z_at_value

from eagle_tools import constants as C

from illustris_tools import *

f_b_TNG = 0.0486/0.3089

sim = 'TNG100-1'
snapnum = 99

file_loc = '/hpcdata7/arijdav1/halo_data/illustris_'

fname = file_loc+sim+'_snap%03d'%(snapnum)+'.hdf5'
filestr = sim+'_snap%03d'%(snapnum)+'.hdf5'

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


gas = TNGSnapshot(sim = sim, snapnum = snapnum)
stars = TNGSnapshot(sim = sim, snapnum = snapnum)
blackholes = TNGSnapshot(sim = sim, snapnum = snapnum)

this_z = gas.z
rhocrit = Planck13.critical_density(this_z).value

z_300Myr = z_at_value(Planck13.lookback_time, 0.3 * u.Gyr)
aexp_300Myr = 1./(1.+z_300Myr)

M200 = np.log10(gas.M200 * 1e10) # Log stellar masses
r200 = gas.r200 / 1000. # Mpc (EAGLE units)
groupnumbers = gas.groupnumbers

groupnumbers = groupnumbers[M200>11.5]
r200 = r200[M200>11.5]
M200 = M200[M200>11.5]

M200_cgs = C.unit_mass_cgs*(10.**M200)/1e10

N = len(M200)

outdict = {}
outdict['GroupNumber'] = groupnumbers
outdict['M200'] = M200
outdict['Mstar_30kpc'] = np.zeros(N)
outdict['MeanLogEntropy'] = np.zeros(N)
outdict['MeanHeatingExcisedLogCoolingTime'] = np.zeros(N)
outdict['MeanHeatingAlteredLogCoolingTime'] = np.zeros(N)
outdict['BlackHoleMass'] = np.zeros(N)
outdict['BlackHoleAccretionRate'] = np.zeros(N)
#outdict['E_SN'] = np.zeros(N)
outdict['E_AGN_QuasarMode'] = np.zeros(N)
outdict['E_AGN_RadioMode'] = np.zeros(N)
outdict['BindingEnergy_Baryon'] = np.zeros(N)
outdict['GasFraction'] = np.zeros(N)
outdict['StellarFraction'] = np.zeros(N)
outdict['BaryonFraction'] = np.zeros(N)
outdict['StarFormationRate_30kpc_instantaneous'] = np.zeros(N)
outdict['StarFormationRate_30kpc_integrated300Myr'] = np.zeros(N)



print N,' haloes'


for g in tqdm(range(N)):

    gas.select(groupnumbers[g],parttype=0,region_size='r200')

    mass = gas.load('Masses',cgs=True)
    density = gas.load('Density',cgs=True)
    temp = gas.load_temperature()
    Xe = gas.load('ElectronAbundance')
    H_massfraction = gas.load('GFM_Metals')[:,0]


    ################################################################################################################################
    # Entropy and cooling time

    n_H = density * H_massfraction/C.m_H_cgs # convert into nH cm^-3
    n_e = n_H*Xe # electron density in cm^-3

    particle_S_over_kB = temp/np.power(n_e,2./3.)

    T200 = C.G_cgs*M200_cgs[g]*0.59*C.m_p_cgs/(2.*C.boltzmann_cgs*r200[g]*C.unit_length_cgs)

    S_200_over_kB = T200/(200.*C.f_b_universal_planck*np.power(rhocrit/(1.14*C.m_p_cgs),2./3.))
    entropy = particle_S_over_kB/S_200_over_kB

    outdict['MeanLogEntropy'][g] = np.sum(mass*np.log10(entropy))/np.sum(mass)

    cooling_rate = gas.load('GFM_CoolingRate') * n_H**2 * mass/density # in erg s^-1

    # Assume mu=0.59
    t_cool = (((3./2.)*C.boltzmann_cgs*temp)/cooling_rate) * mass/(0.59*C.m_p_cgs)
    t_cool /= C.Gyr_s # convert to Gyr

    cooling = np.where(t_cool>0.)[0]

    heated = np.where(t_cool<0.)[0]

    t_cool_noheating = deepcopy(t_cool)
    t_cool_noheating[heated] = np.amax(t_cool)

    plt.figure()
    plt.hist(np.log10(t_cool[cooling]),log=True,histtype='step',bins=100)
    plt.show()

    outdict['MeanHeatingExcisedLogCoolingTime'][g] = np.sum(mass[cooling]*np.log10(t_cool[cooling]))/np.sum(mass[cooling])
    outdict['MeanHeatingAlteredLogCoolingTime'][g] = np.sum(mass[cooling]*np.log10(t_cool_noheating[cooling]))/np.sum(mass[cooling])


    ################################################################################################################################
    # Feedback energy fraction

    # The binding energy budget for baryons [erg]
    outdict['BindingEnergy_Baryon'][g] = f_b_TNG * (3./5.) * C.G_cgs * M200_cgs[g]**2 / (r200[g]*C.unit_length_cgs)

    # E_AGN from the BH snapshot data
    try:
        blackholes.select(groupnumbers[g],parttype=5,region_size='r200')

        BH_masses = blackholes.load('BH_Mass',cgs=True)

        mostmassive = np.argmax(BH_masses)

        outdict['BlackHoleMass'][g] = BH_masses[mostmassive]
        outdict['BlackHoleAccretionRate'][g] = blackholes.load('BH_Mdot',cgs=True)[mostmassive]

        outdict['E_AGN_QuasarMode'][g] = blackholes.load('BH_CumEgyInjection_QM',cgs=True)[mostmassive]
        outdict['E_AGN_RadioMode'][g] = blackholes.load('BH_CumEgyInjection_RM',cgs=True)[mostmassive]
    except:
        print 'BH exception raised'

    # # Energy liberated through star formation in the central (erg)
    # # CAN'T DO - ENERGY LIBERATED IS METALLICITY DEPENDENT

    # stars.select(groupnumbers[g],parttype=4,region_size='r200')

    # initialmass = stars.load('GFM_InitialMass')


    # ParticleData.select(groupnumbers[g],parttype=4) 
    # star_sgn = ParticleData.load('SubGroupNumber')

    # fth = ParticleData.load('Feedback_EnergyFraction')[star_sgn==0]
    # initialmass = ParticleData.load('InitialMass')[star_sgn==0]

    # # Total energy liberated (ever)
    # E_SN[g] = np.sum(initialmass * fth * C.unit_mass_cgs * C.SN_erg_per_g)

    ################################################################################################################################
    # Baryon fractions

    stars.select(groupnumbers[g],parttype=4,region_size='r200')
    mass_stars = stars.load('Masses',cgs=True)

    total_gasmass = np.sum(mass)
    total_starmass = np.sum(mass_stars)

    outdict['GasFraction'][g] = (total_gasmass/M200_cgs[g])/f_b_TNG
    outdict['BaryonFraction'][g] = ((total_gasmass+total_starmass)/M200_cgs[g])/f_b_TNG
    outdict['StellarFraction'][g] = (total_starmass/M200_cgs[g])/f_b_TNG


    ################################################################################################################################
    # SFR in a 30kpc aperture (integrated-if poss- and instantaneous)

    # Instantaneous
    gas.select(groupnumbers[g],parttype=0,region_size=30.)
    outdict['StarFormationRate_30kpc_instantaneous'][g] = np.sum(gas.load('StarFormationRate'))

    # Integrated
    stars.select(groupnumbers[g],parttype=4,region_size=30.)
    formationtime = stars.load('GFM_StellarFormationTime')
    initialmass = stars.load('GFM_InitialMass')

    formed_300Myr = np.where(formationtime>aexp_300Myr)[0]
    outdict['StarFormationRate_30kpc_integrated300Myr'][g] = np.sum(initialmass[formed_300Myr] * 1e10)/(3e8) # in Msol yr^-1

    ################################################################################################################################
    # Stellar mass

    stars.select(groupnumbers[g],parttype=4,region_size=30.)
    outdict['Mstar_30kpc'] = np.sum(stars.load('Masses'))

print 'Saving...'

for key in outdict.keys():
    f.create_dataset(key,data=outdict[key])

f.close()




