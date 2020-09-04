import sys
import os
from abc import ABCMeta, abstractmethod

# Required so numpy parallelization doesn't conflict with multiprocessing
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from .pacomath import *
from multiprocessing import Pool

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


class PACO:
    """
    This class implements the bulk of the PACO algorithm as described by
    Flasseur et al 2018.
    In general, the idea is to take in an ADI stack of images and statistically
    determine if there is a signal above the background in each 'patch' of the image.
    This is done by tracing the ark of the hypothesized planet through the stack,
    and comparing this set of patches to a set consisting of background only.
    This is done for each pixel (or sub-pixel) location in.
    The output is a signal-to-noise and/or a flux map over the field of view.

    What is actually returned is two matrices, a and b. b/sqrt(a) is the SNR map,
    while b/a is the flux map. For the correct flux the iterative procedure should
    be used.
    """

    def __init__(self,
                 image_stack=None,
                 image_file=None,
                 angles=None,
                 psf=None,
                 psf_rad=4,
                 px_scale=1,
                 res_scale=1,
                 verbose=False):
        """
        PACO Parent Class Constructor

        Parameters
        -----------------------------
        image_stack : arr
            Array of 2D science frames taken in pupil tracking/ADI mode
        angles : arr
            List of differential angles between each frame.
        psf : arr
            2D PSF image. Will be normalized such that the peak is 1.
        psf_rad : float
            Radius of PSF in as. Default values give 4px radius.
        px_scale : float
            arcsec per pixel.  Default values give 4px radius.
        res_scale : float
            Scaling for sub/super pixel resolution for PACO to run on
        verbose : bool
            Sets level of printed outputs
        """

        self.m_verbose = verbose
        # Science image setup
        self.m_im_stack = None
        if image_stack is not None:
            self.m_im_stack = np.array(image_stack)
        if image_file is not None:
            self.m_im_stack = np.load(image_file).astype(np.float16)
        self.m_nFrames = 0
        self.m_width = 0
        self.m_height = 0
        if self.m_im_stack is not None:
            self.m_nFrames = self.m_im_stack.shape[0]
            self.m_width = self.m_im_stack.shape[2]
            self.m_height = self.m_im_stack.shape[1]
        else:
            print("Must input image data!")
            sys.exit(1)
        # Parallactic angles
        if angles is not None:
            self.m_angles = angles
        else:
            print("Please add the parallactic angle data to the input image port.")
            sys.exit(1)
        # Pixel scaling
        self.m_pxscale = px_scale
        self.m_scale = res_scale
        self.m_rescaled = False

        # PSF setup
        self.m_psf_rad = int(psf_rad/px_scale)
        if psf is not None:
            # How do we want to deal with stacks of psfs? Median? Just take the first one?
            # Ideally if nPSFs = nImages, use each for each. Need to update!
            if len(psf.shape) > 2:
                psf = psf[0]
            self.m_psf = psf/np.nanmax(psf) #/np.sum(psf)# HOW SHOULD THE PSF BE NORMALISED!?!?!?
            mask = createCircularMask(self.m_psf.shape, self.m_psf_rad)
            self.m_psf_area = self.m_psf[mask].shape[0]
        else:
            print("Please input a PSF template!")
            sys.exit(1)
        self.m_pwidth = 2*int(self.m_psf_rad) + 3
        # Diagnostics
        if self.m_verbose:
            print("---------------------- ")
            print("Summary of PACO setup: \n")
            print("Image Cube shape = " + str(self.m_im_stack.shape))
            print("PIXSCALE = " + str(self.m_pxscale).zfill(6))
            print("PSF |  Area  |  Rad   |  Width | ")
            print("    |   " + str(self.m_psf_area).zfill(2) + \
                  "   |   " + str(self.m_psf_rad).zfill(2) + \
                  "   |  " + str(self.m_psf.shape[0]).zfill(3) + "   | ")
            print("Patch width: " + str(self.m_pwidth))
            print("---------------------- \n")
            print(psf_rad, px_scale)
            sys.stdout.flush()

    def __del__(self):
        """
        Destructor. Does nothing right now.
        """

        if self.m_verbose:
            print("Finished processing with PACO!")
            
    @abstractmethod
    def PACOCalc(self,
                 phi0s,
                 cpu=1):
        """
        This function is algorithm dependant, and sets up the actual calculation process.
        """
    
    def PACO(self,
             cpu=1):
        """
        PACO
        This function wraps the actual PACO algorithm, setting up the pixel coordinates
        that will be iterated over. The output will probably be changes to output the
        true SNR map.

        Parameters
        ------------
        cpu : int>=1
            Number of processers to use
        """

        if (self.m_width * self.m_scale)%2 != 0:
            # This ensures that we're dealing with an even number of pixels.
            # Necessary so that we can evenly divide and rotate about the center.
            # TODO: Shouldn't be necessary?
            self.m_im_stack = self.m_im_stack[0:self.m_nFrames,
                                              0:self.m_width -1,
                                              0:self.m_height-1]
            self.m_width -= 1
            self.m_height -= 1
        if not self.m_rescaled:
            self.rescaleAll()
        if self.m_verbose:
            print("---------------------- ")
            print("Using " + str(cpu) + " processor(s).")
            print("Rescaled Image Cube shape: " + str(self.m_im_stack.shape))
            print("Rescaled PSF:")
            print("PSF |  Area  |  Rad   |  Width | ")
            print("    |   " + str(self.m_psf_area).zfill(2) + \
                  "   |   " + str(self.m_psf_rad).zfill(2) + \
                  "   |  " + str(self.m_pwidth).zfill(3) + "   | ")
            print("---------------------- \n")
        # Setup pixel coordinates
        x, y = np.meshgrid(np.arange(0, self.m_height),
                           np.arange(0, self.m_width))
        phi0s = np.column_stack((x.flatten(), y.flatten()))
        # Compute a,b
        a, b = self.PACOCalc(np.array(phi0s), cpu=cpu)
        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a, (self.m_height, self.m_width))
        b = np.reshape(b, (self.m_height, self.m_width))
        return a, b

    """
    Utility Functions
    """
    # Set number of pixels in a patch
    def setPatchSize(self, npx):
        """
        Set number of pixels in a patch
        """

        self.m_psf_area = npx
        self.m_psf_rad = np.sqrt(npx)/np.pi
    def getPatchSize(self):
        """
        Access to number of pixels
        """

        return self.m_psf_area

    # Set the image stack to be processed
    def setImageSequence(self, imgs):
        """
        Provide a 3D image array to process
        """

        self.m_im_stack = np.array(imgs)
        self.m_nFrames = self.m_im_stack.shape[0]
        self.m_width = self.m_im_stack.shape[2]
        self.m_height = self.m_im_stack.shape[1]
    def getImageSequence(self):
        """
        Access to the data
        """
        
        return self.m_im_stack
    def rescaleImageSequence(self, scale):
        """
        Rescale each image in the stack by the scaling factor
        """

        new_stack = []
        for i, img in enumerate(self.m_im_stack):
            new_stack.append(resizeImage(img, scale))
        self.m_im_stack = np.array(new_stack)
        self.m_width = int(self.m_width * scale)
        self.m_height = int(self.m_height * scale)

    # Set the template PSF
    def setPSF(self, psf):
        """
        Read in the PSF template
        """

        self.m_psf = psf
        self.m_pwidth = self.m_psf.shape[0]
        mask = createCircularMask(self.m_psf.shape, self.m_psf_rad)
        self.m_psf_area = len(mask[mask])
    def getPSF(self):
        """
        Access the PSF
        """

        return self.m_psf

    # Set parallactic angles
    def setAngles(self, angles):
        """
        Set the rotation angle for each frame
        """

        self.m_angles = angles

    def getPatch(self, px, width, mask=None):
        """
        Gets patch at given pixel px with size k for the current img sequence

        Parameters
        --------------
        px : (int,int)
            Pixel coordinates for center of patch
        width : int
            width of a square patch to be masked
        mask : arr
            Circular mask to select a round patch of pixels, which is then
            flattened into a 1D array.

        """

        k = int(width/2)
        if width%2 != 0:
            k2 = k+1
        else:
            k2 = k
        nx, ny = np.shape(self.m_im_stack[0])[:2]
        if px[0]+k2 > nx or px[0]-k < 0 or px[1]+k2 > ny or px[1]-k < 0:
            #print("pixel out of range")
            #return np.full((self.m_im_stack.shape[0],self.m_psf_area),np.nan)
            return None
        if mask is not None:
            patch = np.array([self.m_im_stack[i][int(px[0])-k:int(px[0])+k2,
                                                 int(px[1])-k:int(px[1])+k2][mask] for i in range(len(self.m_im_stack))])
        else:
            patch = np.array([self.m_im_stack[i][int(px[0])-k:int(px[0])+k2,
                                                 int(px[1])-k:int(px[1])+k2] for i in range(len(self.m_im_stack))])
        return patch
        
    def getPatchFast(self, px, width):
        """
        Gets patch at given pixel px with size k for the current img sequence

        Parameters
        --------------
        px : (int,int)
            Pixel coordinates for center of patch
        width : int
            width of a square patch to be masked
        mask : arr
            Circular mask to select a round patch of pixels, which is then
            flattened into a 1D array.

        """
        mask = createCircularMask((self.m_im_stack.shape[1],
                                   self.m_im_stack.shape[2]),
                                   radius = self.m_psf_rad,
                                   center = px)
        k = int(width/2)
        if width%2 != 0:
            k2 = k+1
        else:
            k2 = k
        nx, ny = np.shape(self.m_im_stack[0])[:2]
        if px[0]+k2 > nx or px[0]-k < 0 or px[1]+k2 > ny or px[1]-k < 0:
            #print("pixel out of range")
            #return np.full((self.m_im_stack.shape[0],self.m_psf_area),np.nan)
            return None
        patch = np.zeros((self.m_nFrames, self.m_psf_area))

        for i in range(patch.shape[0]):
            temp = mask*self.m_im_stack[i]
            patch[i] = temp[mask].flatten()
        #if mask is not None:
        #    patch = np.array([self.m_im_stack[i][int(px[0])-k:int(px[0])+k2,
        #                                         int(px[1])-k:int(px[1])+k2][mask] for i in range(len(self.m_im_stack))])
        #else:
        #    patch = np.array([self.m_im_stack[i][int(px[0])-k:int(px[0])+k2,
        #                                         int(px[1])-k:int(px[1])+k2] for i in range(len(self.m_im_stack))])
        return patch
        
    def setScale(self, scale):
        """
        Set subpixel scaling factor
        """

        self.m_scale = scale
        self.m_rescaled = False

    def rescaleAll(self):
        """
        Rescale each image in the stack by the the scaling factor
        A scaling factor of 2 will turn a 100x100 image into 200x200
        """
 
        if self.m_scale == 1:
            if self.m_verbose:
                print("Scale is 1, no scaling applied.")
            self.m_rescaled = True
            return
        
        #try:
        #    assert (self.m_width*self.m_scale).is_integer()
        #except AssertionError:
        #    print("Cannot rescale image, please change the scale or use the full image")
        #    sys.exit(2)
        #try:
        #    assert (self.m_height*self.m_scale).is_integer()
        #except AssertionError:
        #    print("Cannot rescale image, please change the scale or use the full image")
        #    sys.exit(2)   
        #
        self.rescaleImageSequence(self.m_scale)

        self.m_pxscale = self.m_pxscale/self.m_scale
        self.m_psf_rad = int(self.m_psf_rad*self.m_scale)
        if self.m_psf is not None:
            self.m_psf = resizeImage(self.m_psf, self.m_scale)
        mask = createCircularMask(self.m_psf.shape, self.m_psf_rad)
        self.m_psf_area = self.m_psf[mask].shape[0]
        self.m_pwidth = 2*int(self.m_psf_rad) + 3
        self.m_rescaled = True

    """
    Math Functions
    """
    def modelFunction(self, n, model, params):
        """
        This function is deprecated in favour of directly supplying
        a PSF through the Pynpoint database.
        
        Parameters
        -------------
        n : float
            If using the psfTemplateModel function, the mean
        model : dnc
            numpy statistical model (need to import numpy module for this)
        **kwargs: dict
            additional arguments for model
        """

        if self.m_psf:
            return self.m_psf
        if model is None:
            print("Please input either a 2D PSF or a model function.")
            sys.exit(1)
        else:
            if model.__name__ == "psfTemplateModel":
                try:
                    self.m_psf = model(n, params)
                    return self.m_psf
                except ValueError:
                    print("Fix template size")
            self.m_psf = model(n, params)
            return self.m_psf



    def al(self, hfl, Cfl_inv):
        """
        a_l

        The sum of a_l is the inverse of the variance of the background at the given pixel
        """

        a = np.sum(np.array([np.dot(hfl[i], np.dot(Cfl_inv[i], hfl[i]).T) \
                             for i in range(len(hfl))]), axis=0)
        return a

    def bl(self, hfl, Cfl_inv, r_fl, m_fl):
        """
        b_l

        The sum of b_l is the flux estimate at the given pixel.
        """

        b = np.sum(np.array([np.dot(np.dot(Cfl_inv[i], hfl[i]).T, (r_fl[i][i]-m_fl[i]))\
                             for i in range(len(hfl))]), axis=0)
        return b

    """
    FluxPACO
    """
    def fluxEstimate(self,
                     phi0s,
                     eps=0.1,
                     initial_est=[0.0]):
        """
        Unbiased estimate of the flux of a source located at p0
        The estimate of the flux is given by ahat * h, where h is the normalised PSF template
        Not sure how the cost function from Flasseur factors in, or why this is so unstable

        Parameters
        ------------
        p0 : arr
            List of locations of sources to compute unbiased flux estimate
        eps : float
            Precision requirement for iteration (0,1)
        params : dict
            Dictionary describing the analytic template model, or containing the PSF
        initial_est : float
            Initial estimate of the flux at p0
        scale : float
            Resolution scaling
        model_name: method
            Name of the template for the off-axis PSF
        """

        if self.m_verbose:
            print("Estimating the flux")
            print(phi0s)
            print(initial_est)
        print("Computing unbiased flux estimate...")

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        mask = createCircularMask((self.m_pwidth, self.m_pwidth), radius=self.m_psf_rad)
        psf_mask = createCircularMask(self.m_psf.shape, radius=self.m_psf_rad)
        h = np.zeros((self.m_nFrames, self.m_psf_area)) # The off axis PSF at each point

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        # 2d selection of pixels around a given point
        patch = np.zeros((self.m_nFrames, self.m_nFrames, self.m_psf_area))
        x, y = np.meshgrid(np.arange(0, int(self.m_height)),
                           np.arange(0, int(self.m_width)))
        ests = []
        for i, p0 in enumerate(phi0s):
            angles_px = getRotatedPixels(x, y, p0, self.m_angles)

            # Fill patches and signal template
            for l, ang in enumerate(angles_px):
                # Get the column of patches at this point
                patch[l] = self.getPatch(ang, self.m_pwidth, mask)
                h[l] = self.m_psf[psf_mask]

            # the mean of a temporal column of patches at each pixel
            m = np.zeros((self.m_nFrames, self.m_psf_area))
            # the inverse covariance matrix at each point
            Cinv = np.zeros((self.m_nFrames, self.m_psf_area, self.m_psf_area))

            # Unbiased flux estimation
            ahat = initial_est[i]/2.0
            aprev = 9999.0 # Arbitrary large value so that the loop will run
            while np.abs(ahat - aprev) > (ahat * eps):
                a = 0.0
                b = 0.0
                for l in range(self.m_nFrames):
                    m[l], Cinv[l] = self.iterStep(ahat, patch[l], h[l])
                a = self.al(h, Cinv)
                b = self.bl(h, Cinv, patch, m)
                aprev = ahat
                ahat = max(b, 0.0)/a
            ests.append(ahat)
        return ests

    def iterStep(self, est, patch, model):
        """
        Compute the iterative estimates for the mean and inverse covariance

        Parameters
        ----------
        est : float
            Current estimate for the magnitude of the flux
        patch : arr
            Column of patches about p0
        model : arr
            Template for PSF
        """

        if patch is None:
            return None, None
        T = patch.shape[0]

        unbiased = np.array([apatch - est*model for apatch in patch])
        m = np.mean(unbiased, axis=0)
        S = sampleCovariance(unbiased, m, T)
        rho = shrinkageFactor(S, T)
        F = diagSampleCovariance(S)
        C = covariance(rho, S, F)
        Cinv = np.linalg.inv(C)
        return m, Cinv

    def thresholdDetection(self, snr_map, threshold):
        """
        Returns a list of the pixel coordinates of center of signals above a given threshold

        Parameters:
        ------------
        snr_map : arr
            SNR map, b/sqrt(a)
        threshold: float
            Threshold for detection in sigma
        """

        data_max = filters.maximum_filter(snr_map, size=self.m_psf_rad)
        maxima = (snr_map == data_max)
        #data_min = filters.minimum_filter(snr_map,self.m_psf_rad)
        diff = (data_max > threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2
            y.append(y_center)
        return np.array(list(zip(x, y)))

"""
**************************************************
*                                                *
*                                                *
*              Fast PACO/PACitO                  *
*                                                *
*                                                *
**************************************************
"""
class FastPACO(PACO):
    """
    Algorithm Functions
    """

    def PACOCalc(self,
                 phi0s,
                 cpu=1):
        """
        PACOCalc

        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.

        phi0s : int arr
            Array of pixel locations to estimate companion position
        cpu : int >= 1
            Number of cores to use for parallel processing
        """

        npx = len(phi0s)  # Number of pixels in an image
        dim = self.m_width/2

        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        if cpu == 1:
            Cinv, m, h = self.computeStatistics(phi0s)
        else:
            Cinv, m, h = self.computeStatisticsParallel(phi0s, cpu=cpu)

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches

        # 2d selection of pixels around a given point:
        patch = np.zeros((self.m_nFrames, self.m_nFrames, self.m_psf_area))
        mask = createCircularMask((self.m_pwidth, self.m_pwidth), radius=self.m_psf_rad)

        # Currently forcing integer grid, but meshgrid takes floats as arguments...
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        if self.m_verbose:
            print("Running Fast PACO...")
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i, p0 in enumerate(phi0s):
            # Get Angles
            angles_px = getRotatedPixels(x, y, p0, self.m_angles)

            # Ensure within image bounds
            if(int(np.max(angles_px.flatten())) >= self.m_width or \
               int(np.min(angles_px.flatten())) < 0):
                a[i] = np.nan
                b[i] = np.nan
                continue

            # Extract relevant patches and statistics
            Cinlst = []
            mlst = []
            hlst = []
            for l, ang in enumerate(angles_px):
                Cinlst.append(Cinv[int(ang[0]), int(ang[1])])
                mlst.append(m[int(ang[0]), int(ang[1])])
                hlst.append(h[int(ang[0]), int(ang[1])])
                patch[l] = self.getPatchFast(ang, self.m_pwidth)
            #Cinv_arr = np.array(Cinlst)
            #m_arr = np.array(mlst)
            #hl = np.array(hlst)

            #print(Cinlst.shape,mlst.shape,hlst.shape,a.shape,patch.shape)
            # Calculate a and b, matrices
            a[i] = self.al(hlst, Cinlst)
            b[i] = self.bl(hlst, Cinlst, patch, mlst)
        if self.m_verbose:
            print("Done")
        return a, b

    def computeStatistics(self, phi0s):
        """
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack in Serial.

        Parameters
        ---------------
        phi0s : arr
            Array of pixel locations to estimate companion position

        """

        if self.m_verbose:
            print("Precomputing Statistics...")

        mask = createCircularMask((self.m_pwidth, self.m_pwidth), radius=self.m_psf_rad)
        psf_mask = createCircularMask(self.m_psf.shape, radius=self.m_psf_rad)

        # The off axis PSF at each point
        h = np.zeros((self.m_width, self.m_height, self.m_psf_area))

        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        #patch = np.zeros((self.m_nFrames,self.m_psf_area))

        # the mean of a temporal column of patches centered at each pixel
        m = np.zeros((self.m_height, self.m_width, self.m_psf_area))
        # the inverse covariance matrix at each point
        Cinv = np.zeros((self.m_height, self.m_width, self.m_psf_area, self.m_psf_area))

        # *** SERIAL ***
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for p0 in phi0s:
            apatch = self.getPatchFast(p0, self.m_pwidth)
            #if apatch is not None:
            #print(self.m_psf_area,self.m_psf_rad,apatch.shape,h.shape,m.shape,Cinv.shape)
            m[p0[0]][p0[1]], Cinv[p0[0]][p0[1]] = pixelCalc(apatch)
            h[p0[0]][p0[1]] = self.m_psf[psf_mask]
        return Cinv, m, h

    def computeStatisticsParallel(self, phi0s, cpu):
        """
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack in Serial.
        NOTES: This function currently seems slower than computing in serial...

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

        """

        if self.m_verbose:
            print("Precomputing Statistics using %d Processes..."%cpu)
        npx = len(phi0s) # Number of pixels in an image
        mask = createCircularMask((self.m_pwidth, self.m_pwidth), radius=self.m_psf_rad)
        psf_mask = createCircularMask(self.m_psf.shape, radius=self.m_psf_rad)

        # The off axis PSF at each point
        h = np.zeros((self.m_width, self.m_height, self.m_psf_area))

        # the mean of a temporal column of patches at each pixel
        m = np.zeros((self.m_height*self.m_width*self.m_psf_area))
        # the inverse covariance matrix at each point
        Cinv = np.zeros((self.m_height*self.m_width*self.m_psf_area*self.m_psf_area))
        for p0 in phi0s:
            h[p0[0]][p0[1]] = self.m_psf[psf_mask]

        # *** Parallel Processing ***
        #start = time.time()
        arglist = np.array([self.getPatch(p0, self.m_pwidth, mask) for p0 in phi0s])
        p = Pool(processes=cpu)
        data = p.map(pixelCalc, arglist, chunksize=int(npx/cpu))
        p.close()
        p.join()
        ms, cs = [], []
        for d in data:
            if d[0] is None or d[1] is None:
                ms.append(np.full(self.m_psf_area, np.nan))
                cs.append(np.full((self.m_psf_area, self.m_psf_area), np.nan))
            else:
                ms.append(d[0])
                cs.append(d[1])
        ms = np.array(ms)
        cs = np.array(cs)
        m = ms.reshape((self.m_height, self.m_width, self.m_psf_area))
        Cinv = cs.reshape((self.m_height, self.m_width, self.m_psf_area, self.m_psf_area))
        #end = time.time()
        #print("Parallel elapsed",end-start)
        return Cinv, m, h

"""
**************************************************
*                                                *
*                                                *
*                  Full PACO                     *
*                                                *
*                                                *
**************************************************
"""
class FullPACO(PACO):
    """
    Algorithm Functions
    """
    
    def PACOCalc(self,
                 phi0s,
                 cpu=1):
        """
        PACOCalc

        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.

        phi0s : int arr
            Array of pixel locations to estimate companion position
        cpu : int >= 1
            Number of cores to use for parallel processing. Not yet implemented.
        """
   
        npx = len(phi0s)  # Number of pixels in an image
        dim = self.m_width/2

        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        mask = createCircularMask(self.m_psf.shape, radius=self.m_psf_rad)
        h = np.zeros((self.m_nFrames, self.m_psf_area)) # The off axis PSF at each point

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        # 2d selection of pixels around a given point
        patch = np.zeros((self.m_nFrames, self.m_nFrames, self.m_psf_area))

        # the mean of a temporal column of patches at each pixel
        m = np.zeros((self.m_nFrames, self.m_psf_area))
        # the inverse covariance matrix at each point
        Cinv = np.zeros((self.m_nFrames, self.m_psf_area, self.m_psf_area))

        if self.m_verbose:
            print("Running Full PACO...")

        # Set up coordinates so 0 is at the center of the image
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))

        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i, p0 in enumerate(phi0s):
            # Get list of pixels for each rotation angle
            angles_px = getRotatedPixels(x, y, p0, self.m_angles)

            # Iterate over each temporal frame/each angle
            # Same as iterating over phi_l
            for l, ang in enumerate(angles_px):
                # Get the column of patches at this point
                patch[l] = self.getPatch(ang, self.m_pwidth, mask)
                m[l], Cinv[l] = pixelCalc(patch[l])
                h[l] = self.m_psf[mask]

            # Calculate a and b, matrices
            a[i] = self.al(h, Cinv)
            b[i] = self.bl(h, Cinv, patch, m)
        if self.m_verbose:
            print("Done")
        return a, b
