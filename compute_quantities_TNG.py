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

from copy import deepcopy

from multiprocessing import Pool

import astropy.units as u
from astropy.cosmology import Planck13, z_at_value

from eagle_tools import constants as C

from illustris_tools import *

from illustris_python import snapshot as particles 

path.insert(0, '/home/arijdav1/Dropbox/phd/Code/EAGLE_xray/')
import eaglexray_tools as EX

path.insert(0, '/home/arijdav1/Dropbox/phd/Code/morphokinematics/')
from morphokinematicsdiagnostics import *

def apec_cooling_calc(params):
    T,A = params
    return apec_table.total_cooling(T,A)




f_b_TNG = 0.0486/0.3089

sim = 'TNG100-1'
snapnum = 99

TNG_path = '/hpcdata5/simulations/IllustrisTNG/'+sim+'/output/'

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

massmask = np.where(M200>11.5)[0]

groupnumbers = groupnumbers[massmask]
r200 = r200[massmask]
M200 = M200[massmask]

################################################################################################################################
# Match TNG to the DMO run to get a proxy for the intrinsic binding energy

DMO = TNGSnapshot(sim = sim+'-Dark', snapnum = snapnum)

# Don't really need this, it's just to get the length of the SUBFIND lists
ref_Vmax = gas.load_subfind('SubhaloVmax',all_subhalos=True)
# Get the subhalo IDs of all the centrals in our sample
ref_subhaloIDs = np.arange(len(ref_Vmax))[gas.first_subhalo][massmask]

# Load matching files
with h5.File('/hpcdata5/simulations/IllustrisTNG/'+sim+'/postprocessing/subhalo_matching_to_dark.hdf5', 'r') as matchfile:
    
    LHaloTree_matches = np.array(matchfile['Snapshot_%i/SubhaloIndexDark_LHaloTree'%snapnum]) # More rigorous bijective method
    SubLink_matches = np.array(matchfile['Snapshot_%i/SubhaloIndexDark_SubLink'%snapnum]) # Simpler TNG -> Dark only method

DMO_subhaloIDs = LHaloTree_matches[ref_subhaloIDs]

DMO_Vmax = DMO.load_subfind('SubhaloVmax',all_subhalos=True)[DMO_subhaloIDs] * C.unit_velocity_cgs
DMO_hostID = DMO.load_subfind('SubhaloGrNr',all_subhalos=True)[DMO_subhaloIDs]
DMO_M200 = DMO.load_FOF('Group_M_Crit200',all_groups=True)[DMO_hostID] / DMO.h
DMO_r200 = DMO.load_FOF('Group_R_Crit200',all_groups=True)[DMO_hostID] * DMO.aexp/DMO.h
DMO_r500 = DMO.load_FOF('Group_R_Crit500',all_groups=True)[DMO_hostID] * DMO.aexp/DMO.h # Use this for binding energy later
DMO_V200 = np.sqrt(C.G_cgs*DMO_M200*C.unit_mass_cgs/(DMO_r200*C.unit_length_cgs/1000.))
DMO_Vmax_ov_V200 = DMO_Vmax/DMO_V200

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

groupnumbers = np.delete(groupnumbers,failures)
M200 = np.delete(M200,failures)
r200 = np.delete(r200,failures)
DMO_Vmax_ov_V200 = np.delete(DMO_Vmax_ov_V200,failures)
DMO_groupnumbers = np.delete(DMO_groupnumbers,failures)
DMO_r500 = np.delete(DMO_r500,failures)
ref_subhaloIDs = np.delete(ref_subhaloIDs,failures)


################################################################################################################################

M200_cgs = C.unit_mass_cgs*(10.**M200)/1e10

N = len(M200)

outdict = {}
outdict['GroupNumber'] = groupnumbers
outdict['Vmax/V200'] = DMO_Vmax_ov_V200
outdict['M200'] = M200
outdict['r200'] = r200
outdict['Mstar_30kpc'] = np.zeros(N)
outdict['MeanLogEntropy'] = np.zeros(N)
outdict['MedianEntropy'] = np.zeros(N)
outdict['MeanHeatingExcisedLogCoolingTime'] = np.zeros(N)
outdict['MeanHeatingAlteredLogCoolingTime'] = np.zeros(N)
outdict['IanCoolingTime_r200'] = np.zeros(N)
outdict['MedianCoolingTime_r200'] = np.zeros(N)
outdict['IanCoolingTime_0p3r200'] = np.zeros(N)
outdict['MedianCoolingTime_0p3r200'] = np.zeros(N)
outdict['BlackHoleMass'] = np.zeros(N)
outdict['BlackHoleAccretionRate'] = np.zeros(N)
outdict['BlackHoleEddingtonRate'] = np.zeros(N)
outdict['E_SN'] = np.zeros(N)
outdict['E_AGN_QuasarMode'] = np.zeros(N)
outdict['E_AGN_RadioMode'] = np.zeros(N)
outdict['BindingEnergy_Baryon'] = np.zeros(N)
outdict['ParticleBindingEnergy'] = np.zeros(N)
outdict['ParticleBindingEnergy_noSF'] = np.zeros(N)
outdict['GasFraction'] = np.zeros(N)
outdict['GasFraction_0p3r200'] = np.zeros(N)
outdict['StellarFraction'] = np.zeros(N)
outdict['BaryonFraction'] = np.zeros(N)
outdict['ISM_mass'] = np.zeros(N)
outdict['StarFormationRate_30kpc_instantaneous'] = np.zeros(N)
outdict['StarFormationRate_30kpc_integrated300Myr'] = np.zeros(N)
outdict['BindingEnergy_DMO_r500'] = np.zeros(N)
outdict['Lx_r200'] = np.zeros(N)
outdict['KappaCoRot'] = np.zeros(N)


print N,' haloes'


# For X-ray calculation
apec_table = EX.APEC_table('ROSAT')


for g in tqdm(range(N)):

    if 'testing' in argv:
        if g<2000:
            continue

    ################################################################################################################################
    # DMO binding energy (?)

    DMO.select(DMO_groupnumbers[g],parttype=1,region_size=DMO_r500[g]*DMO.aexp/DMO.h)

    PE = DMO.load('Potential',cgs=True)
    vel = DMO.load('Velocities',cgs=True)

    KE = 0.5 * DMO.masstable[1] * C.unit_mass_cgs * np.einsum('...j,...j->...',vel,vel)

    outdict['BindingEnergy_DMO_r500'][g] = np.sum(PE) + np.sum(KE) # in erg


    ################################################################################################################################

    # ISM mass
    gas.select(groupnumbers[g],parttype=0,region_size=30.)

    SFR = gas.load('StarFormationRate')
    mass = gas.load('Masses',cgs=False)*1e10

    outdict['ISM_mass'][g] = np.sum(mass[SFR>0.])

    ################################################################################################################################

    gas.select(groupnumbers[g],parttype=0,region_size='r200')

    SFR = gas.load('StarFormationRate',cgs=True)
    sfmask = np.where(SFR==0.)[0]
    mass = gas.load('Masses',cgs=True)[sfmask]
    density = gas.load('Density',cgs=True)[sfmask]
    temp = gas.load_temperature()[sfmask]
    Xe = gas.load('ElectronAbundance')[sfmask]
    internalenergy = gas.load('InternalEnergy',cgs=True)[sfmask]

    mass_abunds, num_abunds, myXe, Xi, mu = gas.load_abundances()
    mass_abunds = mass_abunds[sfmask]
    num_abunds = num_abunds[sfmask]
    Xi = Xi[sfmask]
    mu = mu[sfmask]

    H_massfraction = mass_abunds[:,0]


    ################################################################################################################################
    # Entropy and cooling time

    n_H = density * H_massfraction/C.m_H_cgs # convert into nH cm^-3
    n_e = n_H*Xe # electron density in cm^-3

    particle_S_over_kB = temp/np.power(n_e,2./3.)

    T200 = C.G_cgs*M200_cgs[g]*0.59*C.m_p_cgs/(2.*C.boltzmann_cgs*r200[g]*C.unit_length_cgs)

    S_200_over_kB = T200/(200.*C.f_b_universal_planck*np.power(rhocrit/(1.14*C.m_p_cgs),2./3.))
    entropy = particle_S_over_kB/S_200_over_kB

    outdict['MeanLogEntropy'][g] = np.sum(mass*np.log10(entropy))/np.sum(mass)
    outdict['MedianEntropy'][g] = np.median(entropy)

    cooling_rate = -1.* gas.load('GFM_CoolingRate')[sfmask] * n_H**2 * mass/density # in erg s^-1

    # Assume mu=0.59
    # t_cool = (((3./2.)*C.boltzmann_cgs*temp)/cooling_rate) * mass/(0.59*C.m_p_cgs)
    t_cool = internalenergy*mass/cooling_rate
    t_cool /= C.Gyr_s # convert to Gyr

    cooling = np.where(t_cool>0.)[0]

    heated = np.where(t_cool<0.)[0]

    t_cool_noheating = deepcopy(t_cool)
    t_cool_noheating[heated] = np.amax(t_cool)

    # print np.log10(np.median(t_cool)), np.log10((np.sum(internalenergy*mass)/np.sum(cooling_rate))/C.Gyr_s)


    outdict['MedianCoolingTime_r200'][g] = np.median(t_cool)
    # outdict['MeanHeatingExcisedLogCoolingTime'][g] = np.sum(mass[cooling]*np.log10(t_cool[cooling]))/np.sum(mass[cooling])
    # outdict['MeanHeatingAlteredLogCoolingTime'][g] = np.sum(mass[cooling]*np.log10(t_cool_noheating[cooling]))/np.sum(mass[cooling])
    outdict['IanCoolingTime_r200'][g] = np.sum(internalenergy*mass)/np.sum(cooling_rate)


    ################################################################################################################################
    # X-ray luminosity in r200

    # temp_indices = apec_table.assign_curves(temp)

    # params = zip(temp_indices,num_abunds)
    # pool = Pool(64)
    # particle_cooling = pool.map(apec_cooling_calc,params)
    # pool.close()
    # pool.join()
    
    # outdict['Lx_r200'][g] = np.sum(EX.calculate_Lx(particle_cooling,density,mass,mu,Xe,Xi)) # density and mass are already in cgs units

    ################################################################################################################################
    # Feedback energy fraction

    # The binding energy budget for baryons [erg]
    outdict['BindingEnergy_Baryon'][g] = f_b_TNG * (3./5.) * C.G_cgs * M200_cgs[g]**2 / (r200[g]*C.unit_length_cgs)

    PE = gas.load('Potential',cgs=True)
    vel = gas.load('Velocities',cgs=True)
    p_mass = gas.load('Masses',cgs=True)

    KE = 0.5 * p_mass * np.einsum('...j,...j->...',vel,vel)

    outdict['ParticleBindingEnergy'][g] = np.sum(PE*1e10*p_mass) + np.sum(KE) # in erg

    PE = PE[sfmask]
    vel = vel[sfmask,:]

    KE = 0.5 * mass * np.einsum('...j,...j->...',vel,vel)

    outdict['ParticleBindingEnergy_noSF'][g] = np.sum(PE*1e10*mass) + np.sum(KE) # in erg


    # E_AGN from the BH snapshot data
    try:
        blackholes.select(groupnumbers[g],parttype=5,region_size='r200')

        BH_masses = blackholes.load('BH_Mass',cgs=True)

        mostmassive = np.argmax(BH_masses)

        outdict['BlackHoleMass'][g] = BH_masses[mostmassive]
        outdict['BlackHoleAccretionRate'][g] = blackholes.load('BH_Mdot',cgs=True)[mostmassive]
        outdict['BlackHoleEddingtonRate'][g] = blackholes.load('BH_MdotEddington',cgs=True)[mostmassive]

        outdict['E_AGN_QuasarMode'][g] = blackholes.load('BH_CumEgyInjection_QM',cgs=True)[mostmassive]
        outdict['E_AGN_RadioMode'][g] = blackholes.load('BH_CumEgyInjection_RM',cgs=True)[mostmassive]
    except:
        print 'BH exception raised'

    # # Energy liberated through star formation in the central (erg)
    # # CAN'T DO - ENERGY LIBERATED IS METALLICITY DEPENDENT

    load_stars = particles.loadSubhalo(TNG_path,snapnum,ref_subhaloIDs[g],4,fields=['GFM_Metallicity','GFM_InitialMass'])

    # Total energy liberated (ever)
    outdict['E_SN'][g] = np.sum(load_stars['GFM_InitialMass'] * C.unit_mass_cgs * SN_erg_per_g(load_stars['GFM_Metallicity']))

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
    # Gas fraction and cooling time in a smaller aperture

    gas.select(groupnumbers[g],parttype=0,region_size=0.3*r200[g]*1000.)
    
    SFR = gas.load('StarFormationRate',cgs=True)
    sfmask = np.where(SFR==0.)[0]
    mass = gas.load('Masses',cgs=True)[sfmask]
    density = gas.load('Density',cgs=True)[sfmask]
    # temp = gas.load_temperature()[sfmask]  
    internalenergy = gas.load('InternalEnergy',cgs=True)[sfmask]
    mass_abunds, num_abunds, myXe, Xi, mu = gas.load_abundances()
    mass_abunds = mass_abunds[sfmask]
    H_massfraction = mass_abunds[:,0]
    n_H = density * H_massfraction/C.m_H_cgs # convert into nH cm^-3
    cooling_rate = -1.* gas.load('GFM_CoolingRate')[sfmask] * n_H**2 * mass/density # in erg s^-1

    # t_cool = (((3./2.)*C.boltzmann_cgs*temp)/cooling_rate) * mass/(0.59*C.m_p_cgs)
    t_cool = internalenergy*mass/cooling_rate
    t_cool /= C.Gyr_s # convert to Gyr

    outdict['MedianCoolingTime_0p3r200'][g] = np.median(t_cool)

    outdict['IanCoolingTime_0p3r200'][g] = np.sum(internalenergy*mass)/np.sum(cooling_rate)

    outdict['GasFraction_0p3r200'][g] = (np.sum(mass)/M200_cgs[g])/f_b_TNG


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

    #stars.select(groupnumbers[g],parttype=4,region_size=30.)
    mass = stars.load('Masses')
    outdict['Mstar_30kpc'][g] = np.sum(mass)

    ################################################################################################################################
    # Kinematics

    coords = stars.load('Coordinates')
    velocity = stars.load('Velocities')
    bindingenergy = np.ones(len(stars.particle_selection)) # binding energy is only important for computing circularity so it's unimportant here

    coords = center_and_unloop(coords,stars.centre,BoxL=stars.boxsize)

    # Kinematical diagnostics
    try:
        outdict['KappaCoRot'][g], kappa_weighted, discfrac, orbi, vrotsig, delta, zaxis_temp,Momentum_temp = kinematics_diagnostics(coords,mass,velocity,bindingenergy,aperture=30.,CoMvelocity=True)
    
    except ValueError: # Catch any weird errors
        outdict['KappaCoRot'][g] = -100.




print 'Saving...'

for key in outdict.keys():
    f.create_dataset(key,data=outdict[key])

f.close()




