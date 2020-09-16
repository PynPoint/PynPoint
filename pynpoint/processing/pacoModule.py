"""
Wrapper for the PACO algorithm implementaion for Pynpoint
"""
import sys
import os
import math
import time
import warnings

import multiprocessing as mp
from typing import Tuple, List
# Required to make parallel processing work
# Else numpy uses multiple processes, which conflicts
# with the multiprocessing module.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from scipy.interpolate import griddata
from typeguard import typechecked

from pynpoint.util.paco import PACO, FastPACO, FullPACO
from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import create_mask
from pynpoint.util.module import progress


class PACOModule(ProcessingModule):
    """
    Pipeline module for generating an SNR map of an image stack using PACO.
    """

    __author__ = 'Evert Nasedkin'
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
                 verbose: bool = False) -> None:
        """
        Constructor of PACOModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that contains the stack with images.
        psf_in_tag : str
            Tag of the database entry that contains the reference PSF that is used as fake
            planet. Can be either a single image (2D) or a cube (3D) with the dimensions
            equal to *image_in_tag*.
        snr_out_tag : str
            Tag of the database entry that contains the SNR map computed using one of the
            PACO algorithms.
        flux_out_tag : str
            Tag of the database entry that contains the list the unbiased flux estimation
            computed using one of the PACO algorithms
        psf_rad : float
            Radius around the psf to use as a 'patch' in arcseconds.
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
        
        super(PACOModule, self).__init__(name_in)
        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        self.m_snr_out_port = self.add_output_port(snr_out_tag)

        if flux_calc:
            self.m_flux_out_port = self.add_output_port(flux_out_tag)
            self.m_source_flux_out_port = self.add_output_port('source_' + flux_out_tag)
            self.m_source_posn_out_port = self.add_output_port('posn_' + flux_out_tag)

        else:
            self.m_flux_out_port = None
            self.m_source_flux_out_port = None
            self.m_source_posn_out_port = None

        self.m_algorithm = algorithm
        self.m_flux_calc = flux_calc
        self.m_scale = scaling
        self.m_psf_rad = psf_rad
        self.m_eps = flux_prec
        self.m_threshold = threshold
        self.m_verbose = verbose
    @typechecked
    def run(self) -> None:
        """
        Run function for PACO.

        Returns
        -------
        NoneType
            None
        """

        # Hardware settings
        cpu = self._m_config_port.get_attribute('CPU')
        # Read in science frames and psf model
        # Should add existance checks
        images = self.m_image_in_port.get_all()

        # Read in parallactic angles, and use the first frame as the 0 reference.
        angles = self.m_image_in_port.get_attribute("PARANG")

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        psf = self.m_psf_in_port.get_all()

        # Setup PACO
        if self.m_algorithm == "fastpaco":
            fp = FastPACO(image_stack=images,
                          angles=angles,
                          psf=psf,
                          psf_rad=self.m_psf_rad,
                          px_scale=pixscale,
                          res_scale=self.m_scale,
                          verbose=self.m_verbose)
        elif self.m_algorithm == "fullpaco":
            fp = FullPACO(image_stack=images,
                          angles=angles,
                          psf=psf,
                          psf_rad=self.m_psf_rad,
                          px_scale=pixscale,
                          res_scale=self.m_scale,
                          verbose=self.m_verbose)
        else:
            print("Please input either 'fastpaco' or 'fullpaco' for the algorithm")

        sys.stdout.write("Running PACOModule...\r")
        sys.stdout.flush()
        # Run PACO
        a, b = fp.PACO(cpu=cpu)
        snr = b/np.sqrt(a)
        flux = b/a
        # Iterative, unbiased flux estimation
        if self.m_flux_calc:
            phi0s = fp.thresholdDetection(snr, self.m_threshold)
            init = np.array([flux[int(phi0[0]), int(phi0[1])] for phi0 in phi0s])
            ests = np.array(fp.fluxEstimate(phi0s, self.m_eps, init))

        # Output

        # Set all creates new data set/overwrites
        # Try MEMORY keyword -
        # database dataset (eg images)
        # Static attributes eg pixel scale
        # Non-static - stored separately
        # set_attr() for output port
        self.m_snr_out_port.set_all(snr, data_dim=2)
        self.m_snr_out_port.close_port()
        if self.m_flux_calc:
            self.m_flux_out_port.set_all(flux, data_dim=2)
            self.m_flux_out_port.close_port()
            self.m_source_flux_out_port.set_all(ests, data_dim=1)
            self.m_source_flux_out_port.close_port()
            self.m_source_posn_out_port.set_all(np.array(phi0s), data_dim=2)
            self.m_source_posn_out_port.close_port()
        sys.stdout.write("\rRunning PACOModule... [DONE]\n")
        sys.stdout.flush()
