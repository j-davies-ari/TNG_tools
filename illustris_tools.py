# -*- coding: utf-8 -*-

import numpy as np
from illustris_python import groupcat as groups 
from illustris_python import snapshot as particles 
import h5py as h5
from sys import exit

class TNGSnapshot(object):
    '''
    Note this class behaves slightly differently to EagleSnapshot. Here, 'select' only loads in particles bound to the FoF halo you choose.
    I don't know if this misses anything out, but I don't think it will, it's SUBFIND that misses outflows etc.
    '''

    def __init__(self, sim = 'TNG100-1', snapnum = 99):

        self.sim = sim
        self.snapnum = snapnum
        self.sim_path = '/hpcdata5/simulations/IllustrisTNG/'+sim+'/output/'
        self.snapfile = self.sim_path + 'snapdir_%03d/snap_%03d.0.hdf5'%((snapnum,snapnum)) # the first chunk, for getting conversion factors later

        # Load volume information
        header = groups.loadHeader(self.sim_path,self.snapnum)
        self.h = header['HubbleParam']
        self.z = header['Redshift']
        self.aexp = 1./(1.+self.z)
        self.boxsize = header['BoxSize']/self.h # physical units

        with h5.File(self.snapfile, 'r') as f:
            # particleheader = f['Header'].attrs['MassTable']
            # h_conversion_factor = temp_data.attrs['h_scaling']
            self.masstable = f['Header'].attrs['MassTable']/self.h


        # FOF quantities

        FOF_quantities = ['GroupFirstSub','Group_M_Crit200','Group_R_Crit200']
        FOF_load = groups.loadHalos(self.sim_path,self.snapnum,fields=FOF_quantities)

        first_subhalo = FOF_load['GroupFirstSub']
        M200 = FOF_load['Group_M_Crit200'] / self.h
        r200 = FOF_load['Group_R_Crit200'] * self.aexp/self.h
        groupnumbers = np.arange(len(M200))+1

        non_empty = np.where(first_subhalo>=0)[0]

        self.M200 = M200[non_empty]
        self.r200 = r200[non_empty]
        self.groupnumbers = groupnumbers[non_empty]
        first_subhalo = first_subhalo[non_empty]

        # SUBFIND quantities
        self.subfind_centres = groups.loadSubhalos(self.sim_path,self.snapnum,fields='SubhaloPos')[first_subhalo,:] * self.aexp/self.h

        self.have_run_select = False


        # # Load all particle coordinates for making masks with get_mask

        # fields = ['Coordinates']
        # self.pos = particles.loadSubset(self.sim_path,self.snapnum,self.parttype,fields=fields) * self.aexp/self.h

    def select(self,groupnumber,parttype=0,region_size='r200'): # Region size is in pkpc

        self.groupnumber = groupnumber
        self.parttype = parttype

        location = np.where(self.groupnumbers==groupnumber)[0]

        assert len(location) == 1,'Multiple central subhaloes found for group number'

        # Get the positions of all particles in this FOF halo

        self.pos = particles.loadHalo(self.sim_path,self.snapnum,self.groupnumber-1,self.parttype,fields='Coordinates') * self.aexp/self.h

        # Get the centre of potential from SUBFIND
        centre = self.subfind_centres[location,:]
        #code_centre = centre * self.h/self.aexp # convert to h-less comoving code units

        # If the region size hasn't been given, set it to r200 (this is the default)
        if region_size == 'r200':
            region_size = self.r200[location]

        #code_region_size = region_size * self.h/self.aexp # convert to h-full comoving code units

        # Now we just need to establish which of the particles we loaded in are within the spherical region.
        
        if np.any((self.boxsize-centre)<region_size) or np.any(centre<region_size):
            # Wrap the box
            self.pos -= centre
            self.pos[self.pos[:,0]<(-1.*self.boxsize/2.),0] += self.boxsize
            self.pos[self.pos[:,1]<(-1.*self.boxsize/2.),1] += self.boxsize
            self.pos[self.pos[:,2]<(-1.*self.boxsize/2.),2] += self.boxsize
            self.pos[self.pos[:,0]>self.boxsize/2.,0] -= self.boxsize
            self.pos[self.pos[:,1]>self.boxsize/2.,1] -= self.boxsize
            self.pos[self.pos[:,2]>self.boxsize/2.,2] -= self.boxsize
        else:
            self.pos -= centre

        # Create a mask to the radial region we want, for future use.
        r2 = np.einsum('...j,...j->...',self.pos,self.pos) # get the radii from the centre

        self.particle_selection = np.where(r2<region_size**2)[0]

        self.centre = centre

        self.have_run_select = True


        
    
    def load(self,quantity,verbose=False,cgs=False):

        # Get our factors of h and a to convert to physical units
        with h5.File(self.snapfile, 'r') as f:
            temp_data = f['/PartType%i/%s'%((self.parttype,quantity))]
            h_conversion_factor = temp_data.attrs['h_scaling']
            aexp_conversion_factor = temp_data.attrs['a_scaling']
            CGS_conversion_factor = temp_data.attrs['to_cgs']

        if verbose:
            print 'Loading ',quantity
            print 'h exponent = ',h_conversion_factor
            print 'a exponent = ',aexp_conversion_factor
            print 'CGS conversion factor = ',CGS_conversion_factor

        # Load in the quantity

        loaded_data = particles.loadHalo(self.sim_path,self.snapnum,self.groupnumber-1,self.parttype,fields=quantity)
        loaded_data = loaded_data[self.particle_selection]

        if cgs and CGS_conversion_factor != 0.:
            # Recast as CGS numbers can be huge
            loaded_data = loaded_data.astype(np.float64) * np.float64(CGS_conversion_factor)

        return loaded_data * np.power(self.h,h_conversion_factor) * np.power(self.aexp,aexp_conversion_factor)


    def load_temperature(self):

        m_p_cgs = 1.6726219e-24
        boltzmann_cgs = np.float64(1.38064852e-16)

        internalenergy = self.load('InternalEnergy',cgs=True)
        H_massfraction = self.load('GFM_Metals')[:,0]
        electron_abundance = self.load('ElectronAbundance')

        mu = (4.*m_p_cgs)/(1.+3.*H_massfraction+4.*H_massfraction*electron_abundance)

        return ((5./3.)-1) * mu * internalenergy/boltzmann_cgs


    def load_abundances(self):

        '''
        Returns arrays of mass abundance and number ratio, as well as X_e, X_i and mu.
        '''

        abunds = np.zeros((len(self.particle_selection),11))

        metals = self.load('GFM_Metals')
        abunds[:,:8] = metals[:,:8]
        abunds[:,8] = abunds[:,7]*0.6054160
        abunds[:,9] = abunds[:,7]*0.0941736
        abunds[:,10] = metals[:,8]

        masses_in_u = np.array([1.00794,4.002602,12.0107,14.0067,15.9994,20.1797,24.3050,28.0855,32.065,40.078,55.845])
        atomic_numbers = np.array([1.,2.,6.,7.,8.,10.,12.,14.,16.,20.,26.])
        Xe = np.ones(len(abunds[:,0])) # Initialise values for hydrogen
        Xi = np.ones(len(abunds[:,0]))
        mu = np.ones(len(abunds[:,0]))*0.5
        num_ratios = np.zeros(np.shape(abunds))
        for col in range(len(abunds[0,:])): # convert mX/mtot to mX/mH
            num_ratios[:,col] = abunds[:,col] / abunds[:,0]
        for element in range(len(abunds[0,:])-1):
            mu += num_ratios[:,element+1]/(1.+atomic_numbers[element+1])
            num_ratios[:,element+1] *= masses_in_u[0]/masses_in_u[element+1] # convert mX/mH to nX/nH (H automatically done before)
            Xe += num_ratios[:,element+1]*atomic_numbers[element+1] # Assuming complete ionisation
            Xi += num_ratios[:,element+1]

        return abunds, num_ratios, Xe, Xi, mu

def SN_erg_per_g(Z):

    Z_w = 0.002
    gamma_wz = 2
    f_wz = 0.25
    ebar_w = 3.6
    N_SNII = 0.0118

    m_sol_cgs = 1.989e33

    e_w = ebar_w * (f_wz + (1.-f_wz)/((1.+(Z/Z_w))**gamma_wz)) * N_SNII * 1e51

    return e_w / m_sol_cgs







if __name__ == '__main__':

    snapshot = TNGSnapshot() # Defaults are TNG100-1 simulation, z=0 (this is the only one we have downloaded)

    snapshot.select(2586) # select a ~MW mass halo by it's FOF group number (NOT its index, i.e. start at 1 not 0)
    # Defaults are parttype=0, region size r200

    # Alternatively you can specify a particle type and a RADIAL region size in pkpc (eg here for getting M*_30kpc)
    # snapshot.select(2586,parttype=4,region_size=30.)

    # Now you can load in the properties of particles in this halo
    coords = snapshot.load('Coordinates') # h and a corrections are done automatically
    print coords

    # You can get the output in cgs units with cgs=True (default is False) e.g.:
    # density = snapshot.load('Coordinates',cgs=True)
    # I recommend doing this cause Illustris units are a bit different to EAGLE. 

    # One last quirk - you can't directly load Temperature from Illustris, you have to compute it from other things
    # This class has a function specifically for loading temperature
    print snapshot.load_temperature()

    # The nice thing about this module is that you can loop over haloes really easily and quickly, for example..
    # Let's make dot plots of the stellar contents of a few MW-mass haloes in the x-y plane

    import matplotlib.pyplot as plt

    for gn in np.arange(2586,2590):

        snapshot.select(gn,parttype=4)
        coords = snapshot.load('Coordinates') # output in pkpc
        plt.figure(figsize=(8,6))
        plt.scatter(coords[:,0],coords[:,1],s=1,c='k')
        plt.show()





