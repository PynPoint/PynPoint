"""
This module implements ALGORITHM 2 from the PACO paper. 
Covariance and mean statistics are computed and stored 
before running the algorithm, but do not have subpixel
accuracy.
"""
from .pacomath import *
from .paco import PACO
from multiprocessing import Pool

import sys,os
import time

#Pacito
class FastPACO(PACO):
    """
    Algorithm Functions
    """      
    def PACOCalc(self,
                 phi0s,
                 cpu = 1):
        """
        PACO_calc
        
        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.
        
        phi0s : int arr
            Array of pixel locations to estimate companion position
        params: dict
            Dictionary of parameters about the psf, containing either the width
            of a gaussian distribution, or a label 'psf_template'
        scale : float
            Resolution scaling
        model_name: str
            Name of the template for the off-axis PSF
        cpu : int >= 1
            Number of cores to use for parallel processing
        """      
        npx = len(phi0s)  # Number of pixels in an image
        dim = self.m_width/2
        try:
            assert (dim).is_integer()
        except AssertionError:
            print("Image cannot be properly rotated.")
            sys.exit(1)
        dim = int(dim)
        
        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        if cpu == 1:
            Cinv,m,h = self.computeStatistics(phi0s)
        else:
            Cinv,m,h = self.computeStatisticsParallel(phi0s,cpu = cpu)

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((self.m_nFrames,self.m_nFrames,self.m_psf_area)) # 2d selection of pixels around a given point
        mask =  createCircularMask(self.m_psf.shape,radius = self.m_psf_rad)

        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))    

        print("Running PACO...")
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            # Get Angles
            angles_px = getRotatedPixels(x,y,p0,self.m_angles)

            # Ensure within image bounds
            if(int(np.max(angles_px.flatten()))>=self.m_width or \
               int(np.min(angles_px.flatten()))<0):
                a[i] = np.nan
                b[i] = np.nan
                continue

            # Extract relevant patches and statistics
            Cinlst = []
            mlst = []
            hlst = []
            for l,ang in enumerate(angles_px):
                Cinlst.append(Cinv[int(ang[0])][int(ang[1])])
                mlst.append(m[int(ang[0])][int(ang[1])])
                hlst.append(h[int(ang[0])][int(ang[1])])
                patch[l] = self.getPatch(ang, self.m_pwidth, mask)
            Cinv_arr = np.array(Cinlst)
            m_arr   = np.array(mlst)
            hl   = np.array(hlst)

            #print(Cinlst.shape,mlst.shape,hlst.shape,a.shape,patch.shape)
            # Calculate a and b, matrices
            a[i] = self.al(hlst, Cinlst)
            b[i] = self.bl(hlst, Cinlst, patch, mlst)
        print("Done")
        return a,b

    def computeStatistics(self, phi0s):
        """    
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack in Serial.
        Parameters
        ---------------
        phi0s : int arr
            Array of pixel locations to estimate companion position
        params: dict
            Dictionary of parameters about the psf, containing either the width
            of a gaussian distribution, or a label 'psf_template'
        model_name: str
            Name of the template for the off-axis PSF
        """
    
        print("Precomputing Statistics...")
        npx = len(phi0s)       
        dim = int(self.m_width/2)

        mask =  createCircularMask(self.m_psf.shape,radius = self.m_psf_rad)
        if self.m_psf_area != len(mask[mask]):
            self.m_psf_area = len(mask[mask])
        # The off axis PSF at each point
        h = np.zeros((self.m_height,self.m_width,self.m_psf_area)) 

        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((self.m_nFrames,self.m_psf_area))

        # the mean of a temporal column of patches centered at each pixel
        m     = np.zeros((self.m_height,self.m_width,self.m_psf_area)) 
        # the inverse covariance matrix at each point
        Cinv  = np.zeros((self.m_height,self.m_width,self.m_psf_area,self.m_psf_area)) 

        # *** SERIAL ***
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for p0 in phi0s:
            apatch = self.getPatch(p0,self.m_pwidth,mask)
            m[p0[0]][p0[1]],Cinv[p0[0]][p0[1]] = pixelCalc(apatch)           
            h[p0[0]][p0[1]] = self.m_psf[mask]
        return Cinv,m,h

    def computeStatisticsParallel(self, phi0s, cpu):
        """    
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack in Serial.
        Parameters
        ---------------
        phi0s : int arr
            Array of pixel locations to estimate companion position
        params: dict
            Dictionary of parameters about the psf, containing either the width
            of a gaussian distribution, or a label 'psf_template'
        scale : float
            Resolution scaling
        model_name: str
            Name of the template for the off-axis PSF
        cpu : int
            Number of processors to use
            
        NOTES:
        This function currently seems slower than computing in serial...
        """
    
        print("Precomputing Statistics using %d Processes...",cpu)
        npx = len(phi0s)           # Number of pixels in an image      
        dim = int(self.m_width/2)
        mask =  createCircularMask(self.m_psf.shape,radius = self.m_psf_rad)
        if self.m_psf_area != len(mask[mask]):
            self.m_psf_area = len(mask[mask])
        # The off axis PSF at each point
        h = np.zeros((self.m_height,self.m_width,self.m_psf_area)) 

        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((self.m_nFrames,self.m_psf_area))           
        # the mean of a temporal column of patches at each pixel
        m     = np.zeros((self.m_height*self.m_width*self.m_psf_area)) 
        # the inverse covariance matrix at each point
        Cinv  = np.zeros((self.m_height*self.m_width*self.m_psf_area*self.m_psf_area)) 
        for p0 in phi0s:
            h[p0[0]][p0[1]] = self.m_psf[mask]
                
        # *** Parallel Processing ***
        #start = time.time()
        arglist = [np.copy(np.array(self.getPatch(p0, k, mask))) for p0 in phi0s]
        p = Pool(processes = cpu)
        data = p.map(pixelCalc, arglist, chunksize = int(npx/cpu))
        p.close()
        p.join()
        ms,cs = [],[]
        for d in data:
            ms.append(d[0])
            cs.append(d[1])
        ms = np.array(ms)
        cs = np.array(cs)  
        m = ms.reshape((self.m_height,self.m_width,self.m_psf_area))
        Cinv = cs.reshape((self.m_height,self.m_width,self.m_psf_area,self.m_psf_area))
        #end = time.time()
        #print("Parallel elapsed",end-start)
        return Cinv,m,h
