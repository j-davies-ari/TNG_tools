# -*- coding: utf-8 -*-

import numpy as np
import h5py as h5
import copy
import eagle as E
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sys import argv, exit, path
from tqdm import tqdm
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable

import safe_colours
safe_colours = safe_colours.initialise()
col_dict = safe_colours.distinct_named()
rainbow_cmap = safe_colours.colourmap('rainbow')

from illustris_tools import *

path.insert(0,'/home/arijdav1/Dropbox/phd/Code/EAGLE_xray/')
from eaglexray_tools import *

path.insert(0,'/home/arijdav1/Dropbox/phd/Code/EAGLE_tools/')
from eagle_tools import *
C = constants()


eagle_location = '/hpcdata5/simulations/EAGLE/L0100N1504/REFERENCE/data/'
f_b_TNG = 0.0486/0.3089

# Set mass selections
closest_masses = np.array([12.,12.5,13.,14.])

# Initialise snapshots

eagle_snap = EagleSnapshot()

tng_snap = TNGSnapshot()

# Find the right haloes in both boxes.

# EAGLE
eagle_M200 = np.log10(np.array(E.readArray("SUBFIND_GROUP",eagle_location,'028_z000p000','FOF/Group_M_Crit200'))*1e10)
eagle_groupnumbers = np.arange(len(eagle_M200)) + 1

eagle_mask = np.where(eagle_M200>11.5)[0]
eagle_M200 = eagle_M200[eagle_mask]
eagle_groupnumbers = eagle_groupnumbers[eagle_mask]

eagle_sort = np.argsort(eagle_M200)
eagle_M200 = eagle_M200[eagle_sort]
eagle_groupnumbers = eagle_groupnumbers[eagle_sort]

eagle_locations = searchsort_locate(eagle_M200,closest_masses)
eagle_groupnumbers = eagle_groupnumbers[eagle_locations]
eagle_M200 = eagle_M200[eagle_locations]

# IllustrisTNG
tng_M200 = np.log10(tng_snap.M200 * 1e10) # Log stellar masses
tng_groupnumbers = tng_snap.groupnumbers

tng_mask = np.where(tng_M200>11.5)[0]
tng_M200 = tng_M200[tng_mask]
tng_groupnumbers = tng_groupnumbers[tng_mask]

tng_sort = np.argsort(tng_M200)
tng_M200 = tng_M200[tng_sort]
tng_groupnumbers = tng_groupnumbers[tng_sort]

tng_locations = searchsort_locate(tng_M200,closest_masses)
tng_groupnumbers = tng_groupnumbers[tng_locations]
tng_M200 = tng_M200[tng_locations]

print 'Target masses: ',closest_masses
print 'EAGLE masses: ',eagle_M200
print 'IllustrisTNG masses: ',tng_M200



for h, halomass in enumerate(closest_masses):

    if 'rhoT' in argv:

        fig, (eax,iax) = plt.subplots(1,2,figsize=(16,8),sharey=True)
        fig.subplots_adjust(wspace=0)

        # EAGLE
        eagle_snap.select(eagle_groupnumbers[h],parttype=0,region_size='r200')

        eagle_density = eagle_snap.load('Density') * C.unit_density_cgs
        eagle_H_abund = eagle_snap.load_abundances()[0][:,0]
        eagle_nH = np.log10(eagle_density * eagle_H_abund/C.m_H_cgs)
        eagle_temp = np.log10(eagle_snap.load('Temperature'))

        eagle_map = eax.hexbin(eagle_nH,eagle_temp,gridsize=200,bins='log',cmap='CMRmap_r')
        eax.set_ylabel(r'$\log(T)\,[{\rm K}]$',fontsize=16)
        eax.set_xlabel(r'$\log(n_{\rm H})\,[{\rm cm}^{-3}]$',fontsize=16)

        # IllustrisTNG
        tng_snap.select(tng_groupnumbers[h],parttype=0,region_size='r200')

        tng_density = tng_snap.load('Density',cgs=True)
        tng_H_abund = tng_snap.load('GFM_Metals')[:,0]
        tng_nH = np.log10(tng_density * tng_H_abund/C.m_H_cgs)
        tng_temp = np.log10(tng_snap.load_temperature())

        tng_map = iax.hexbin(tng_nH,tng_temp,gridsize=200,bins='log',cmap='CMRmap_r')
        iax.set_xlabel(r'$\log(n_{\rm H})\,[{\rm cm}^{-3}]$',fontsize=16)

        eax.annotate(r'$\mathrm{EAGLE}$'+'\n'+r'$\log(M_{200})[{\rm M_{\odot}}]=%.1f$'%(halomass),xy=(-1.5,6.7),fontsize=16)
        iax.annotate(r'$\mathrm{IllustrisTNG}$'+'\n'+r'$\log(M_{200})[{\rm M_{\odot}}]=%.1f$'%(halomass),xy=(-1.5,6.7),fontsize=16)

        maps = [eagle_map,tng_map]

        for a, ax in enumerate([eax,iax]):
            ax.set_ylim(3.,9.)
            ax.set_xlim(-6.8,1.8)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top",size="5%",pad=0.)
            col = plt.colorbar(maps[a], cax=cax, orientation='horizontal')
            col.ax.xaxis.set_ticks_position('top')
            col.set_label(r'$\log(N)$',labelpad=-55.,fontsize=16) 

        plt.savefig('/home/arijdav1/Dropbox/phd/figures/paper_plots/scatter_plots/morphology/TNG100-1/testing/rhoT_M%.1f.png'%(halomass))


    elif 'coolingtime' in argv:

        fig, ax = plt.subplots(1,figsize=(8,6))

        # EAGLE
        eagle_snap.select(eagle_groupnumbers[h],parttype=0,region_size='r200')

        eagle_mass = eagle_snap.load('Mass') * C.unit_mass_cgs
        eagle_density = eagle_snap.load('Density') * C.unit_density_cgs
        eagle_H_abund = eagle_snap.load_abundances()[0][:,0]
        eagle_nH = eagle_density * eagle_H_abund/C.m_H_cgs
        eagle_temp = eagle_snap.load('Temperature')
        try:
            eagle_num_ratios = eagle_snap.load_abundances()[1]
        except:
            continue

        cloudy = CLOUDY_highcadence(redshift=0.)
        tng_luminosity = cloudy.particle_luminosity(eagle_temp,eagle_nH,eagle_num_ratios,eagle_mass,eagle_density)

        eagle_t_cool = (((3./2.)*C.boltzmann_cgs*eagle_temp)/tng_luminosity) * eagle_mass/(0.59*C.m_p_cgs)
        eagle_t_cool /= C.Gyr_s # convert to Gyr

        cooling = np.where(eagle_t_cool>0.)[0]

        ax.hist(np.log10(eagle_t_cool[cooling]),histtype='step',color=col_dict['maroon'],log=True,bins=50)
        

        # IllustrisTNG
        tng_snap.select(tng_groupnumbers[h],parttype=0,region_size='r200')

        tng_mass = tng_snap.load('Masses',cgs=True)
        tng_density = tng_snap.load('Density',cgs=True)
        tng_H_abund = tng_snap.load('GFM_Metals')[:,0]
        tng_nH = tng_density * tng_H_abund/C.m_H_cgs
        tng_temp = tng_snap.load_temperature()
        tng_num_ratios = tng_snap.load_abundances()[1]
        tng_coolingrate_fromsnap = -1. * tng_snap.load('GFM_CoolingRate') * tng_nH**2 * tng_mass/tng_density 

        tng_luminosity = cloudy.particle_luminosity(tng_temp,tng_nH,tng_num_ratios,tng_mass,tng_density)

        tng_t_cool = (((3./2.)*C.boltzmann_cgs*tng_temp)/tng_luminosity) * tng_mass/(0.59*C.m_p_cgs)
        tng_t_cool_fromsnap = (((3./2.)*C.boltzmann_cgs*tng_temp)/tng_coolingrate_fromsnap) * tng_mass/(0.59*C.m_p_cgs)
        tng_t_cool /= C.Gyr_s # convert to Gyr
        tng_t_cool_fromsnap /= C.Gyr_s # convert to Gyr


        cooling_cloudy = np.where(tng_t_cool>0.)[0]
        cooling_snap = np.where(tng_t_cool_fromsnap>0.)[0]

        ax.hist(np.log10(tng_t_cool[cooling_cloudy]),histtype='step',color=col_dict['navy'],log=True,bins=np.linspace(-6.,2.,50))

        ax.hist(np.log10(tng_t_cool_fromsnap[cooling_snap]),histtype='step',color=col_dict['cyan'],log=True,bins=np.linspace(-6.,2.,50))


        ax.set_ylabel(r'$\log(N)$',fontsize=16)
        ax.set_xlabel(r'$\log(t_{\rm cool})\,[{\rm Gyr}]$',fontsize=16)

        plt.savefig('/home/arijdav1/Dropbox/phd/figures/paper_plots/scatter_plots/morphology/TNG100-1/testing/cooling_M%.1f.png'%(halomass))

        # Cooling time rho-T

        fig, (lax,rax) = plt.subplots(1,2,figsize=(16,8),sharey=True)
        fig.subplots_adjust(wspace=0)

        

        cloudy_map = lax.hexbin(np.log10(tng_nH[cooling_cloudy]),np.log10(tng_temp[cooling_cloudy]),C=tng_t_cool[cooling_cloudy], gridsize=200,bins='log',cmap=rainbow_cmap)
        lax.set_xlabel(r'$\log(n_{\rm H})\,[{\rm cm}^{-3}]$',fontsize=16)
        lax.set_ylabel(r'$\log(T)\,[{\rm K}]$',fontsize=16)

        snap_map = rax.hexbin(np.log10(tng_nH[cooling_snap]),np.log10(tng_temp[cooling_snap]),C=tng_t_cool_fromsnap[cooling_snap], gridsize=200,bins='log',cmap=rainbow_cmap)
        rax.set_xlabel(r'$\log(n_{\rm H})\,[{\rm cm}^{-3}]$',fontsize=16)

        lax.annotate(r'$\mathrm{CLOUDY}$'+'\n'+r'$\log(M_{200})[{\rm M_{\odot}}]=%.1f$'%(halomass),xy=(-1.5,6.7),fontsize=16)
        rax.annotate(r'$\mathrm{Snapshot}$'+'\n'+r'$\log(M_{200})[{\rm M_{\odot}}]=%.1f$'%(halomass),xy=(-1.5,6.7),fontsize=16)

        maps = [cloudy_map,snap_map]

        for a, ax in enumerate([lax,rax]):
            ax.set_ylim(3.,9.)
            ax.set_xlim(-6.8,1.8)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top",size="5%",pad=0.)
            col = plt.colorbar(maps[a], cax=cax, orientation='horizontal')
            col.ax.xaxis.set_ticks_position('top')
            col.set_label(r'$\log(t_{\rm cool})$',labelpad=-55.,fontsize=16) 

        plt.savefig('/home/arijdav1/Dropbox/phd/figures/paper_plots/scatter_plots/morphology/TNG100-1/testing/rhoT_coolingtime_M%.1f.png'%(halomass))





    elif 'halo' in argv:

        # EAGLE
        masstable = E.readAttribute('SNAPSHOT', eagle_location, '028_z000p000', "/Header/MassTable") / 0.6777

        eagle_total_mass = 0.

        for ptype in [0,4,5]:
            eagle_snap.select(eagle_groupnumbers[h],parttype=ptype,region_size='r200')

            eagle_total_mass += np.sum(eagle_snap.load('Mass')) *1e10

        eagle_snap.select(eagle_groupnumbers[h],parttype=1,region_size='r200')
        eagle_total_mass += len(eagle_snap.particle_selection) * masstable[1] * 1e10

        print "EAGLE total mass / M200 = ",eagle_total_mass/(10.**eagle_M200[h])


        # IllustrisTNG

        tng_total_mass = 0.

        for ptype in [0,4,5]:
            tng_snap.select(tng_groupnumbers[h],parttype=ptype,region_size='r200')

            tng_total_mass += np.sum(tng_snap.load('Masses')) *1e10

        tng_snap.select(tng_groupnumbers[h],parttype=1,region_size='r200')
        tng_total_mass += len(tng_snap.particle_selection) * tng_snap.masstable[1] * 1e10

        print "IllustrisTNG total mass / M200 = ",tng_total_mass/(10.**tng_M200[h])


    else:
        print 'Please specify what you want to do.'



















