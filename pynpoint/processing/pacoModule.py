import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
from pynpoint.util.paco import PACO
from pynpoint.util.fullpaco import FullPACO
from pynpoint.util.fastpaco import FastPACO
from pynpoint.util.pacomath import *
from pynpoint.core.processing import ProcessingModule


class PACOModule(ProcessingModule):
    def __init__(self,
                 name_in = "paco",
                 image_in_tag = "im_arr",
                 psf_in_tag = None,
                 snr_out_tag = "paco_snr",
                 psf_model = None,
                 angles = None,
                 psf_rad = 4,
                 patch_size = 49,
                 scaling = 1.0,
                 algorithm = "fastpaco",
                 flux_calc = False,
                 psf_params = None,
                 cpu_limit = 1,
                 threshold = 5.0,
                 flux_prec = 0.05
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
            Tag of the database entry that contains the SNR map and unbiased flux estimation
            computed using one of the PACO algorithms
        patch_size : int
            Number of pixels in a circular patch in which the patch covariance is computed
        algorithm : str
            One of 'fastpaco' or 'fullpaco', depending on which PACO algorithm is to be run
        flux_calc : bool
            True if  fluxpaco is to be run, computing the unbiased flux estimation of 
            a set of companions.

        """
        super(PACOModule,self).__init__(name_in)
        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag is not None:
            if psf_in_tag == image_in_tag:
                self.m_psf_in_port = self.m_image_in_port
            else:
                self.m_psf_in_port = self.add_input_port(psf_in_tag)
        else:
            self.m_psf_in_port = None
        self.m_snr_out_port = self.add_output_port(snr_out_tag)
        self.m_algorithm = algorithm
        self.m_patch_size = patch_size
        self.m_angles = angles
        self.m_flux_calc = flux_calc
        self.m_scale = scaling
        self.m_psf_params = psf_params
        self.m_cpu_lim = cpu_limit
        self.m_model_function = psf_odel
        self.m_eps = flux_prec
        self.m_threshold = threshold
    def run(self):
        """
        Run function for PACO
        """
        # Hardware settings
        cpu = self._m_config_port.get_attribute("CPU")
        if cpu>self.m_cpu_lim:
            cpu = self.m_cpu_lim
        # Read in science frames and psf model
        # Should add existance checks
        images = self.m_image_in_port.get_all()
        
        # Read in parallactic angles, and use the first frame as the 0 reference.
        if self.m_angles is not None:
            angles = self.m_angles
        else:
            angles = self.m_image_in_port.get_attribute("PARANG")
        angles = angles - angles[0]

        px_scale = self.m_image_in_port.get_attribute("PIXSCALE")
        if self.m_psf_in_port is not None:
            psf = self.m_psf_in_port.get_all()   
        elif self.m_psf_params is not None:
            psf = None
            
        # Setup PACO
        if self.m_algorithm == "fastpaco":
            fp = paco.processing.fastpaco.FastPACO(image_stack = images,
                                                   angles = angles,
                                                   psf = psf,
                                                   psf_rad = self.m_psf_rad,
                                                   px_scale = self.m_px_scale,
                                                   res_scale = self.m_scale,
                                                   patch_area = self.m_patch_size)
        elif self.m_algorithm == "fullpaco":
            fp = paco.processing.fullpaco.FullPACO(image_stack = images,
                                                   angles = angles,
                                                   psf = psf,
                                                   psf_rad = psf_rad,
                                                   px_scale = px_scale,
                                                   res_scale = self.m_scale,
                                                   patch_area = self.m_patch_size)
        else:
            print("Please input either 'fastpaco' or 'fullpaco' for the algorithm")

        
        # Run PACO
        # SNR = b/sqrt(a)
        # Flux estimate = b/a
        a,b  = fp.PACO(model_params = self.m_psf_params,
                       model_name = self.m_model_function,
                       cpu = cpu)

        snr = b/np.sqrt(a)
        # Iterative, unbiased flux estimation
        if self.m_flux_calc:
            phi0s = fp.thresholdDetection(snr,self.m_threshold)
            est = 0.0
            fp.fluxEstimate(phi0s,self.m_eps,est)
        
        # Output
        
        # Set all creates new data set/overwrites
        # Try MEMORY keyword -
        # database dataset (eg images)
        # Static attributes eg pixel scale
        # Non-static - stored separately
        # set_attr() for output port 
        self.m_snr_out_port.set_all(snr, data_dim=2)
        self.m_snr_out_port.close_port()
