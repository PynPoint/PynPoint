"""
Pipeline modules for estimating detection limits.
"""

from __future__ import absolute_import

import sys
import os
import warnings
import multiprocessing as mp

import numpy as np

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import create_mask
from pynpoint.util.limits import contrast_limit
from pynpoint.util.module import progress
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals


class ContrastCurveModule(ProcessingModule):
    """
    Module to calculate contrast limits by iterating towards a threshold for the false positive
    fraction, with a correction for small sample statistics. Positions are processed in parallel
    if CPU > 1 in the configuration file.
    """

    def __init__(self,
                 name_in="contrast",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 contrast_out_tag="contrast_limits",
                 separation=(0.1, 1., 0.01),
                 angle=(0., 360., 60.),
                 threshold=("sigma", 5.),
                 psf_scaling=1.,
                 aperture=0.05,
                 pca_number=20,
                 cent_size=None,
                 edge_size=None,
                 extra_rot=0.,
                 residuals="median",
                 snr_inject=100.,
                 **kwargs):
        """
        Constructor of ContrastCurveModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that contains the stack with images.
        psf_in_tag : str
            Tag of the database entry that contains the reference PSF that is used as fake planet.
            Can be either a single image or a stack of images equal in size to *image_in_tag*.
        contrast_out_tag : str
            Tag of the database entry that contains the separation, azimuthally averaged contrast
            limits, the azimuthal variance of the contrast limits, and the threshold of the false
            positive fraction associated with sigma.
        separation : tuple(float, float, float)
            Range of separations (arcsec) where the contrast is calculated. Should be specified as
            (lower limit, upper limit, step size). Apertures that fall within the mask radius or
            beyond the image size are removed.
        angle : tuple(float, float, float)
            Range of position angles (deg) where the contrast is calculated. Should be specified as
            (lower limit, upper limit, step size), measured counterclockwise with respect to the
            vertical image axis, i.e. East of North.
        threshold : tuple(str, float)
            Detection threshold for the contrast curve, either in terms of "sigma" or the false
            positive fraction (FPF). The value is a tuple, for example provided as ("sigma", 5.)
            or ("fpf", 1e-6). Note that when sigma is fixed, the false positive fraction will
            change with separation. Also, sigma only corresponds to the standard deviation of a
            normal distribution at large separations (i.e., large number of samples).
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should have a positive value.
        aperture : float
            Aperture radius (arcsec).
        pca_number : int
            Number of principal components used for the PSF subtraction.
        cent_size : float
            Central mask radius (arcsec). No mask is used when set to None.
        edge_size : float
            Outer edge radius (arcsec) beyond which pixels are masked. No outer mask is used when
            set to None. If the value is larger than half the image size then it will be set to
            half the image size.
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).
        residuals : str
            Method used for combining the residuals ("mean", "median", "weighted", or "clipped").
        snr_inject : float
            Signal-to-noise ratio of the injected planet signal that is used to measure the amount
            of self-subtraction.

        Returns
        -------
        NoneType
            None
        """

        super(ContrastCurveModule, self).__init__(name_in)

        if "sigma" in kwargs:
            warnings.warn("The 'sigma' parameter has been deprecated. Please use the 'threshold' "
                          "parameter instead.", DeprecationWarning)

        if "norm" in kwargs:
            warnings.warn("The 'norm' parameter has been deprecated. It is not recommended to "
                          "normalize the images before PSF subtraction.", DeprecationWarning)

        if "accuracy" in kwargs:
            warnings.warn("The 'accuracy' parameter has been deprecated. The parameter is no "
                          "longer required.", DeprecationWarning)

        if "magnitude" in kwargs:
            warnings.warn("The 'magnitude' parameter has been deprecated. The parameter is no "
                          "longer required.", DeprecationWarning)

        if "ignore" in kwargs:
            warnings.warn("The 'ignore' parameter has been deprecated. The parameter is no "
                          "longer required.", DeprecationWarning)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        self.m_contrast_out_port = self.add_output_port(contrast_out_tag)

        self.m_separation = separation
        self.m_angle = angle
        self.m_psf_scaling = psf_scaling
        self.m_threshold = threshold
        self.m_aperture = aperture
        self.m_pca_number = pca_number
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals
        self.m_snr_inject = snr_inject

        if self.m_angle[0] < 0. or self.m_angle[0] > 360. or self.m_angle[1] < 0. or \
           self.m_angle[1] > 360. or self.m_angle[2] < 0. or self.m_angle[2] > 360.:

            raise ValueError("The angular positions of the fake planets should lie between "
                             "0 deg and 360 deg.")

    def run(self):
        """
        Run method of the module. Fake positive companions are injected for a range of separations
        and angles. The magnitude of the contrast is changed stepwise and lowered by a factor 2 if
        needed. Once the fractional accuracy of the false positive fraction threshold is met, a
        linear interpolation is used to determine the final contrast. Note that the sigma level
        is fixed therefore the false positive fraction changes with separation, following the
        Student's t-distribution (Mawet et al. 2014).

        Returns
        -------
        NoneType
            None
        """

        images = self.m_image_in_port.get_all()
        psf = self.m_psf_in_port.get_all()

        if psf.shape[0] != 1 and psf.shape[0] != images.shape[0]:
            raise ValueError('The number of frames in psf_in_tag {0} does not match with the '
                             'number of frames in image_in_tag {1}. The DerotateAndStackModule can '
                             'be used to average the PSF frames (without derotating) before '
                             'applying the ContrastCurveModule.'.format(psf.shape, images.shape))

        cpu = self._m_config_port.get_attribute("CPU")
        parang = self.m_image_in_port.get_attribute("PARANG")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale

        if self.m_edge_size is not None:
            self.m_edge_size /= pixscale

        self.m_aperture /= pixscale

        pos_r = np.arange(self.m_separation[0]/pixscale,
                          self.m_separation[1]/pixscale,
                          self.m_separation[2]/pixscale)

        pos_t = np.arange(self.m_angle[0]+self.m_extra_rot,
                          self.m_angle[1]+self.m_extra_rot,
                          self.m_angle[2])

        if self.m_cent_size is None:
            index_del = np.argwhere(pos_r-self.m_aperture <= 0.)
        else:
            index_del = np.argwhere(pos_r-self.m_aperture <= self.m_cent_size)

        pos_r = np.delete(pos_r, index_del)

        if self.m_edge_size is None or self.m_edge_size > images.shape[1]/2.:
            index_del = np.argwhere(pos_r+self.m_aperture >= images.shape[1]/2.)
        else:
            index_del = np.argwhere(pos_r+self.m_aperture >= self.m_edge_size)

        pos_r = np.delete(pos_r, index_del)

        sys.stdout.write("Running ContrastCurveModule...\r")
        sys.stdout.flush()

        positions = []
        for sep in pos_r:
            for ang in pos_t:
                positions.append((sep, ang))

        # Create a queue object which will contain the results
        queue = mp.Queue()

        result = []
        jobs = []

        working_place = self._m_config_port.get_attribute("WORKING_PLACE")

        # Create temporary files
        tmp_im_str = os.path.join(working_place, "tmp_images.npy")
        tmp_psf_str = os.path.join(working_place, "tmp_psf.npy")

        np.save(tmp_im_str, images)
        np.save(tmp_psf_str, psf)

        mask = create_mask(images.shape[-2:], [self.m_cent_size, self.m_edge_size])

        _, im_res = pca_psf_subtraction(images=images*mask,
                                        angles=-1.*parang+self.m_extra_rot,
                                        pca_number=self.m_pca_number)

        noise = combine_residuals(method=self.m_residuals, res_rot=im_res)

        for i, pos in enumerate(positions):
            process = mp.Process(target=contrast_limit,
                                 args=(tmp_im_str,
                                       tmp_psf_str,
                                       noise,
                                       mask,
                                       parang,
                                       self.m_psf_scaling,
                                       self.m_extra_rot,
                                       self.m_pca_number,
                                       self.m_threshold,
                                       self.m_aperture,
                                       self.m_residuals,
                                       self.m_snr_inject,
                                       pos,
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

            progress(i, len(jobs), "Running ContrastCurveModule...")

        # Send termination sentinel to queue
        queue.put(None)

        while True:
            item = queue.get()

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

        sys.stdout.write("\rRunning ContrastCurveModule... [DONE]\n")
        sys.stdout.flush()

        history = str(self.m_threshold[0])+" = "+str(self.m_threshold[1])

        self.m_contrast_out_port.add_history("ContrastCurveModule", history)
        self.m_contrast_out_port.copy_attributes(self.m_image_in_port)
        self.m_contrast_out_port.close_port()
