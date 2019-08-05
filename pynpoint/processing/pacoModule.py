import sys
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import math
import time
import warnings
import multiprocessing as mp

from typing import Tuple, List

import numpy as np

from scipy.interpolate import griddata
from typeguard import typechecked


from pynpoint.util.paco import PACO,FastPACO,FullPACO
from pynpoint.util.pacomath import *
from pynpoint.util.paco_contrast_limits import paco_contrast_limit

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import create_mask
from pynpoint.util.module import progress

class PACOModule(ProcessingModule):
    @typechecked
    def __init__(self,
                 name_in: str = "paco",
                 image_in_tag: str = "science",
                 psf_in_tag: str = "psf",
                 snr_out_tag: str = "paco_snr",
                 flux_out_tag: str = "paco_flux",
                 psf_rad: float = 4.,
                 scaling: float = 1.,
                 algorithm: str = "fastpaco",
                 flux_calc: bool = True,
                 threshold: float = 5.0,
                 flux_prec: float = 0.05,
                 verbose: bool = True
    ):
        """
        Constructor of PACOModule.
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that contains the stack with images.
        psf_in_tag : str
            Tag of the database entry that contains the reference PSF that is used as fake planet.
            Can be either a single image (2D) or a cube (3D) with the dimensions equal to
            *image_in_tag*.
        snr_out_tag : str
            Tag of the database entry that contains the SNR map computed using one of the 
            PACO algorithms
        flux_out_tag : str
            Tag of the database entry that contains the list the unbiased flux estimation
            computed using one of the PACO algorithms
        psf_rad : float
            Radius around the psf to use as a 'patch'
        scaling : float 
            Greater than 1 to run paco with subpixel positioning, less than one to run on 
            a downscaled resolution for each image.
        algorithm : str
            One of 'fastpaco' or 'fullpaco', depending on which PACO algorithm is to be run
        flux_calc : bool
            True if  fluxpaco is to be run, computing the unbiased flux estimation of 
            a set of companions.
        threshold : float
            Threshold in sigma for a detection to be considered true in the SNR map
        flux_prec : float
            Precision to which the iterative flux calculation must converge.
        verbose : bool
            Sets the level of printed output.
        """
        super(PACOModule,self).__init__(name_in)
        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)
        self.m_snr_out_port = self.add_output_port(snr_out_tag)
        if flux_calc:
            self.m_flux_out_port = self.add_output_port(flux_out_tag)
            self.m_source_flux_out_port = self.add_output_port("source_" + flux_out_tag)
            self.m_source_posn_out_port = self.add_output_port("posn_" + flux_out_tag)

            
        self.m_algorithm = algorithm
        self.m_flux_calc = flux_calc
        self.m_scale = scaling
        self.m_psf_rad = psf_rad
        self.m_eps = flux_prec
        self.m_threshold = threshold
        self.m_verbose = verbose
        
    def run(self):
        """
        Run function for PACO
        """
        # Hardware settings
        cpu = self._m_config_port.get_attribute("CPU")
        # Read in science frames and psf model
        # Should add existance checks
        images = self.m_image_in_port.get_all()
        
        # Read in parallactic angles, and use the first frame as the 0 reference.
        angles = self.m_image_in_port.get_attribute("PARANG")
        angles = angles - angles[0]

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        psf = self.m_psf_in_port.get_all()   
            
        # Setup PACO
        if self.m_algorithm == "fastpaco":
            fp = FastPACO(image_stack = images,
                          angles = angles,
                          psf = psf,
                          psf_rad = self.m_psf_rad,
                          px_scale = pixscale,
                          res_scale = self.m_scale,
                          verbose = self.m_verbose)
        elif self.m_algorithm == "fullpaco":
            fp = FullPACO(image_stack = images,
                          angles = angles,
                          psf = psf,
                          psf_rad = self.m_psf_rad,
                          px_scale = pixscale,
                          res_scale = self.m_scale,
                          verbose = self.m_verbose)
        else:
            print("Please input either 'fastpaco' or 'fullpaco' for the algorithm")

            sys.stdout.write("Running PACOModule...\r")
        sys.stdout.flush()
        # Run PACO
        a,b  = fp.PACO(cpu = cpu)

        snr = b/np.sqrt(a)
        flux = b/a
        # Iterative, unbiased flux estimation
        if self.m_flux_calc:
            phi0s = fp.thresholdDetection(snr,self.m_threshold)
            init = np.array([flux[int(phi0[0]),int(phi0[1])] for phi0 in phi0s])
            ests =  np.array(fp.fluxEstimate(phi0s,self.m_eps,init))
        
        # Output

        # Set all creates new data set/overwrites
        # Try MEMORY keyword -
        # database dataset (eg images)
        # Static attributes eg pixel scale
        # Non-static - stored separately
        # set_attr() for output port 
        self.m_snr_out_port.set_all(snr, data_dim=2)
        self.m_snr_out_port.close_port()
        self.m_flux_out_port.set_all(flux, data_dim=2)
        self.m_flux_out_port.close_port()
        self.m_source_flux_out_port.set_all(ests, data_dim=1)
        self.m_source_flux_out_port.close_port()
        self.m_source_posn_out_port.set_all(np.array(phi0s), data_dim=2)
        self.m_source_posn_out_port.close_port()
        sys.stdout.write("\rRunning PACOModule... [DONE]\n")
        sys.stdout.flush()


class PACOContrastModule(ProcessingModule):
    @typechecked
    def __init__(self,
                 name_in: str = "paco_contrast",
                 image_in_tag: str = "science",
                 psf_in_tag: str = "psf",
                 contrast_out_tag: str = "contrast_out",
                 angle: Tuple[float, float, float] = (0., 360., 60.),
                 separation: Tuple[float, float, float] = (0.1, 1., 0.01),
                 threshold: Tuple[str, float] = ('sigma', 5.),
                 aperture: float = 0.05,
                 snr_inject: float = 100.,
                 extra_rot: float = 0.,
                 psf_scaling: float = 1.0,
                 psf_rad: float = 4,
                 scaling: float = 1.0,
                 algorithm: str = "fastpaco",
                 verbose: bool =  False
    ):
        """
        Constructor of PACOContrastModule.
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that contains the stack with images.
        psf_in_tag : str
            Tag of the database entry that contains the reference PSF that is used as fake planet.
            Can be either a single image (2D) or a cube (3D) with the dimensions equal to
            *image_in_tag*.
        algorithm : str
            One of 'fastpaco' or 'fullpaco', depending on which PACO algorithm is to be run


        """
        super(PACOContrastModule,self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        self.m_contrast_out_port = self.add_output_port(contrast_out_tag)         

        self.m_angle = angle
        if self.m_angle[0] < 0. or self.m_angle[0] > 360. or self.m_angle[1] < 0. or \
           self.m_angle[1] > 360. or self.m_angle[2] < 0. or self.m_angle[2] > 360.:         
            raise ValueError('The angular positions of the fake planets should lie between '
                             '0 deg and 360 deg.')
            
        self.m_separation = separation
        self.m_aperture = aperture
        self.m_threshold = threshold
        self.m_snr_inject = snr_inject
        self.m_extra_rot = extra_rot

        # Scaling notes:
        # self.m_scale        scales the resolution of the images for PACO
        #                     to allow for subpixel template placement.
        # self.m_psf_scaling  scales the brightness of the psf
        # pixscale            Is the pixel size in arcsec
        self.m_algorithm = algorithm
        self.m_scale = scaling
        self.m_psf_rad = psf_rad
        self.m_psf_scaling = psf_scaling
        self.m_verbose = verbose
    @typechecked    
    def run(self) -> None:
        """
        Run method of the module. An artificial planet is injected (based on the noise level) at a
        given separation and position angle. The amount of self-subtraction is then determined and
        the contrast limit is calculated for a given sigma level or false positive fraction. A
        correction for small sample statistics is applied for both cases. Note that if the sigma
        level is fixed, the false positive fraction changes with separation, following the
        Student's t-distribution (see Mawet et al. 2014 for details).

        Returns
        -------
        NoneType
            None
        """

        images = self.m_image_in_port.get_all()
        psf = self.m_psf_in_port.get_all()

        #if psf.shape[0] != 1 and psf.shape[0] != images.shape[0]:
        #    raise ValueError('The number of frames in psf_in_tag {0} does not match with the '
        #                     'number of frames in image_in_tag {1}. The DerotateAndStackModule can '
        #                     'be used to average the PSF frames (without derotating) before '
        #                     'applying the ContrastCurveModule.'.format(psf.shape, images.shape))
        
        cpu = self._m_config_port.get_attribute("CPU")
        parang = self.m_image_in_port.get_attribute("PARANG")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        #if self.m_cent_size is not None:
        #    self.m_cent_size /= pixscale

        #if self.m_edge_size is not None:
        #    self.m_edge_size /= pixscale

        self.m_aperture /= pixscale

        pos_r = np.arange(self.m_separation[0]/pixscale,
                          self.m_separation[1]/pixscale,
                          self.m_separation[2]/pixscale)

        pos_t = np.arange(self.m_angle[0]+self.m_extra_rot,
                          self.m_angle[1]+self.m_extra_rot,
                          self.m_angle[2])

        #if self.m_cent_size is None:
        index_del = np.argwhere(pos_r-self.m_aperture <= 0.)
        #else:
        #   index_del = np.argwhere(pos_r-self.m_aperture <= self.m_cent_size)

        pos_r = np.delete(pos_r, index_del)

        #if self.m_edge_size is None or self.m_edge_size > images.shape[1]/2.:
        index_del = np.argwhere(pos_r+self.m_aperture >= images.shape[1]/2.)
        #else:
        #    index_del = np.argwhere(pos_r+self.m_aperture >= self.m_edge_size)
        pos_r = np.delete(pos_r, index_del)

        sys.stdout.write("Running PACOContrastCurveModule...\r")
        sys.stdout.flush()

        positions = []
        for sep in pos_r:
            for ang in pos_t:
                positions.append((sep, ang))

        working_place = self._m_config_port.get_attribute("WORKING_PLACE")

        # Create temporary files
        tmp_im_str = os.path.join(working_place, "tmp_images.npy")
        tmp_psf_str = os.path.join(working_place, "tmp_psf.npy")

        np.save(tmp_im_str, images)
        np.save(tmp_psf_str, psf)

        #mask = create_mask(images.shape[-2:], [self.m_cent_size, self.m_edge_size])

        #_, im_res = pca_psf_subtraction(images=images*mask,
        #                                angles=-1.*parang+self.m_extra_rot,
        #                                pca_number=self.m_pca_number)

        if self.m_algorithm == "fastpaco":
            fp = FastPACO(image_stack = images,
                          angles = parang,
                          psf = psf,
                          psf_rad = self.m_psf_rad,
                          px_scale = pixscale,
                          res_scale = self.m_scale,
                          verbose = self.m_verbose)
        elif self.m_algorithm == "fullpaco":
            fp = FullPACO(image_stack = images,
                          angles = parang,
                          psf = psf,
                          psf_rad = self.m_psf_rad,
                          px_scale = pixscale,
                          res_scale = self.m_scale,
                          verbose = self.m_verbose)

        # Run PACO
        a,b  = fp.PACO(cpu = cpu)
        noise = b/a
        del fp

        # Create a queue object which will contain the results
        queue = mp.Queue(2*len(positions))
        result = []
        jobs = []
        print("Using",cpu,"cpus for multiprocessing")
        for i, pos in enumerate(positions):
            process = mp.Process(target=paco_contrast_limit,
                                 args=(tmp_im_str,
                                       tmp_psf_str,
                                       noise,
                                       parang,
                                       self.m_psf_rad,
                                       self.m_psf_scaling,
                                       self.m_scale,
                                       pixscale,
                                       self.m_extra_rot,
                                       self.m_threshold,
                                       self.m_aperture,
                                       self.m_snr_inject,
                                       pos,
                                       self.m_algorithm,
                                       queue),
                                 name=(str(os.path.basename(__file__)) + '_radius=' +
                                       str(np.round(pos[0]*pixscale, 1)) + '_angle=' +
                                       str(np.round(pos[1], 1))))

            jobs.append(process)

        for i, job in enumerate(jobs):
            job.start()

            if (i+1)%cpu == 0:
                # Start *cpu* number of processes. Wait for them to finish and start again *cpu*
                # number of processes.
                for k in jobs[i+1-cpu:(i+1)]:
                    k.join()

            elif (i+1) == len(jobs) and (i+1)%cpu != 0:
                # Wait for the last processes to finish if number of processes is not a multiple
                # of *cpu*
                for k in jobs[(i + 1 - (i+1)%cpu):]:
                    k.join()

            progress(i, len(jobs), "Running PACOContrastCurveModule...")

        # Send termination sentinel to queue
        queue.put(None)

        while True:
            item = queue.get()
            sys.stdout.flush()
            if item is None:
                break
            else:
                result.append(item)
        
        os.remove(tmp_im_str)
        os.remove(tmp_psf_str)

        result = np.asarray(result)
        # Sort the results first by separation and then by angle
        indices = np.lexsort((result[:, 1], result[:, 0]))
        result = result[indices]

        result = result.reshape((pos_r.size, pos_t.size, 4))

        mag_mean = np.nanmean(result, axis=1)[:, 2]
        mag_var = np.nanvar(result, axis=1)[:, 2]
        res_fpf = result[:, 0, 3]

        limits = np.column_stack((pos_r*pixscale, mag_mean, mag_var, res_fpf))

        self.m_contrast_out_port.set_all(limits, data_dim=2)

        sys.stdout.write("\rRunning PACOContrastCurveModule... [DONE]\n")
        sys.stdout.flush()

        history = str(self.m_threshold[0])+" = "+str(self.m_threshold[1])

        self.m_contrast_out_port.add_history("PACOContrastCurveModule", history)
        self.m_contrast_out_port.copy_attributes(self.m_image_in_port)
        self.m_contrast_out_port.close_port()
