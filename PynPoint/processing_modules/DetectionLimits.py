"""
Modules for determining detection limits.
"""

import math
import sys
import warnings

import numpy as np

from photutils import aperture_photometry, CircularAperture
from scipy.interpolate import interp1d
from scipy.stats import t

from PynPoint.core import ProcessingModule
from PynPoint.processing_modules import PSFSubtractionModule, FastPCAModule, FakePlanetModule


class ContrastModule(ProcessingModule):
    """
    Module to calculate contrast limits by iterating towards a threshold for the false positive
    fraction, with a correction for small sample statistics.
    """

    def __init__(self,
                 name_in="contrast",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 pca_out_tag="im_pca",
                 contrast_out_tag="contrast_limits",
                 separation=(0.1, 1., 0.01),
                 angle=(0., 360., 60.),
                 magnitude=(7.5, 1.),
                 sigma=5.,
                 accuracy=1e-1,
                 psf_scaling=1.,
                 aperture=0.05,
                 pca_module='PSFSubtractionModule',
                 pca_number=20,
                 mask=0.,
                 extra_rot=0.):
        """
        Constructor of ContrastModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that contains the stack with images.
        :type image_in_tag: str
        :param psf_in_tag: Tag of the database entry that contains the reference PSF that is used
                           as fake planet. Can be either a single image (2D) or a cube (3D) with
                           the dimensions equal to *image_in_tag*.
        :type psf_in_tag: str
        :param pca_out_tag: Tag of the database entry that contains all the residuals of the PSF
                            subtraction for each position and optimization step.
        :type pca_out_tag: str
        :param contrast_out_tag: Tag of the database entry that contains the azimuthally averaged
                                 contrast limits, the azimuthal variance of the contrast limits,
                                 and the threshold of the false positive fraction associated with
                                 sigma.
        :type contrast_out_tag: str
        :param separation: Range of separations (arcsec) where the contrast is calculated. Should
                           be specified as (lower limit, upper limit, step size). Apertures that
                           fall within the mask radius or beyond the image size are removed.
        :type separation: tuple
        :param angle: Range of position angles (deg) where the contrast is calculated. Should be
                      specified as (lower limit, upper limit, step size), measured counterclockwise
                      with respect to the vertical image axis, i.e. East of North.
        :type angle: tuple
        :param magnitude: Initial magnitude value and step size for the fake planet, specified
                          as (planet magnitude, magnitude step size).
        :type magnitude: tuple
        :param sigma: Detection threshold in units of sigma. Note that as sigma is fixed, the
                      confidence level (and false positive fraction) change with separation.
        :type sigma: float
        :param accuracy: Fractional accuracy of the false positive fraction. When the
                         accuracy condition is met, the final magnitude is calculated with a
                         linear interpolation.
        :type accuracy: float
        :param psf_scaling: Additional scaling factor of the planet flux (e.g., to correct for a
                            neutral density filter). Should have a positive value.
        :type psf_scaling: float
        :param aperture: Aperture radius (arcsec) for the calculation of the false positive
                         fraction.
        :type aperture: float
        :param pca_module: Name of the processing module for the PSF subtraction
                           (PSFSubtractionModule or FastPCAModule).
        :type pca_module: str
        :param pca_number: Number of principle components used for the PSF subtraction.
        :type pca_number: int
        :param mask: Mask radius (arcsec) for the PSF subtraction.
        :type mask: float
        :param extra_rot: Additional rotation angle of the images (deg).
        :type extra_rot: float

        :return: None
        """

        super(ContrastModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        self.m_pca_out_port = self.add_output_port(pca_out_tag)
        self.m_contrast_out_port = self.add_output_port(contrast_out_tag)

        self.m_image_in_tag = image_in_tag
        self.m_psf_in_tag = psf_in_tag

        self.m_separation = separation
        self.m_angle = angle
        self.m_sigma = sigma
        self.m_accuracy = accuracy
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_aperture = aperture
        self.m_pca_module = pca_module
        self.m_pca_number = pca_number
        self.m_mask = mask
        self.m_extra_rot = extra_rot

    @staticmethod
    def _false_alarm(image,
                     x_pos,
                     y_pos,
                     size):

        center = (np.size(image, 0)/2., np.size(image, 1)/2.)
        radius = math.sqrt((center[0]-y_pos)**2.+(center[1]-x_pos)**2.)
        num_ap = int(2.*math.pi*radius/size)

        ap_phot = np.zeros(num_ap)
        ap_theta = np.linspace(0, 2.*math.pi, num_ap, endpoint=False)

        for i, theta in enumerate(ap_theta):
            x_tmp = center[1] + (x_pos-center[1])*math.cos(theta) - \
                                (y_pos-center[0])*math.sin(theta)
            y_tmp = center[0] + (x_pos-center[1])*math.sin(theta) + \
                                (y_pos-center[0])*math.cos(theta)

            aperture = CircularAperture((x_tmp, y_tmp), size)
            phot_table = aperture_photometry(image, aperture, method='exact')
            ap_phot[i] = phot_table['aperture_sum']

        t_test = (ap_phot[0] - np.mean(ap_phot[1:])) / \
                 (np.std(ap_phot[1:]) * math.sqrt(1.+1./float(num_ap-1)))

        return 1. - t.cdf(t_test, num_ap-2)

    @staticmethod
    def _student_fpf(sigma,
                     radius,
                     size):

        num_ap = int(2.*math.pi*radius/(2.*size))

        return 1. - t.cdf(sigma, num_ap-2, loc=0., scale=1.)

    def _noise(self,
               sep,
               size):

        psf_sub = FastPCAModule(name_in="pca_contrast",
                                pca_numbers=self.m_pca_number,
                                images_in_tag=self.m_image_in_tag,
                                reference_in_tag=self.m_image_in_tag,
                                res_mean_tag="contrast_res_mean",
                                res_median_tag=None,
                                res_arr_out_tag=None,
                                res_rot_mean_clip_tag=None,
                                extra_rot=self.m_extra_rot,
                                verbose=False)

        psf_sub.connect_database(self._m_data_base)
        psf_sub.run()

        res_input_port = self.add_input_port("contrast_res_mean")
        im_res = res_input_port.get_all()
        im_res = np.squeeze(im_res, axis=0)

        center = (np.size(im_res, 0)/2., np.size(im_res, 1)/2.)

        num_ap = 2.*math.pi*sep/size
        num_ap = num_ap.astype(int)

        noise = np.zeros(np.size(sep))

        for j, radius in enumerate(sep):
            ap_theta = np.linspace(0, 2.*math.pi, num_ap[j], endpoint=False)
            ap_phot = np.zeros(num_ap[j])

            for i, theta in enumerate(ap_theta):
                x_tmp = center[1] + radius*math.cos(theta)
                y_tmp = center[0] + radius*math.sin(theta)

                aperture = CircularAperture((x_tmp, y_tmp), size)
                phot_table = aperture_photometry(im_res, aperture, method='exact')
                ap_phot[i] = phot_table['aperture_sum']

            noise[j] = np.std(ap_phot)

        return noise

    def run(self):
        """
        Run method of the module. Fake positive companions are injected for a range of separations
        and angles. The magnitude of the contrast is changed stepwise and lowered by a factor 2 if
        needed. Once the fractional accuracy of the false positive fraction threshold is met, a
        linear interpolation is used to determine the final contrast. Note that the sigma level
        is fixed therefore the false positive fraction changes with separation, following the
        Student's t-distribution (Mawet et al. 2014).

        :return: None
        """

        if self.m_angle[0] < 0. or self.m_angle[0] > 360. or self.m_angle[1] < 0. or \
           self.m_angle[1] > 360. or self.m_angle[2] < 0. or self.m_angle[2] > 360.:
            raise ValueError("The angular positions of the fake planets should lie between "
                             "0 deg and 360 deg.")

        images = self.m_image_in_port.get_all()
        psf = self.m_psf_in_port.get_all()

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_aperture /= pixscale
        self.m_mask /= pixscale*images.shape[1]

        if psf.ndim == 3 and psf.shape[0] != images.shape[0]:
            warnings.warn('The number of frames in psf_in_tag does not match with the number of '
                          'frames in image_in_tag. Using the mean of psf_in_tag as PSF template.')

        center = np.array([images.shape[2]/2., images.shape[1]/2.])

        pos_r = np.arange(self.m_separation[0]/pixscale,
                          self.m_separation[1]/pixscale,
                          self.m_separation[2]/pixscale)

        pos_t = np.arange(self.m_angle[0]+self.m_extra_rot,
                          self.m_angle[1]+self.m_extra_rot,
                          self.m_angle[2])

        index_del = np.argwhere(pos_r-self.m_aperture <= self.m_mask*images.shape[1])
        pos_r = np.delete(pos_r, index_del)

        index_del = np.argwhere(pos_r+self.m_aperture >= images.shape[1]/2.)
        pos_r = np.delete(pos_r, index_del)

        # noise = self._noise(pos_r, self.m_aperture)

        fake_mag = np.zeros((len(pos_r), len(pos_t)))
        fake_fpf = np.zeros((len(pos_r)))

        count = 1

        for m, sep in enumerate(pos_r):
            fpf_threshold = self._student_fpf(self.m_sigma, sep, self.m_aperture)
            fake_fpf[m] = fpf_threshold

            for n, ang in enumerate(pos_t):
                sys.stdout.write("Processing position " + str(count) + " out of " + \
                      str(np.size(fake_mag))+" ")
                sys.stdout.flush()

                x_fake = center[0] + sep*math.cos(np.radians(ang+90.-self.m_extra_rot))
                y_fake = center[1] + sep*math.sin(np.radians(ang+90.-self.m_extra_rot))

                if n == 0:
                    list_mag = [self.m_magnitude[0]]
                    mag_step = self.m_magnitude[1]

                else:
                    list_mag = [np.mean(fake_mag[m, 0:n])]

                    if n == 1:
                        mag_step = self.m_magnitude[1]
                    else:
                        mag_step = np.std(fake_mag[m, 0:n])
                        if mag_step > 1.:
                            mag_step = 0.2

                list_fpf = []

                iteration = 1

                while True:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                    mag = list_mag[-1]

                    fake_planet = FakePlanetModule(position=(sep*pixscale, ang),
                                                   magnitude=mag,
                                                   psf_scaling=self.m_psf_scaling,
                                                   name_in="fake_planet",
                                                   image_in_tag=self.m_image_in_tag,
                                                   psf_in_tag=self.m_psf_in_tag,
                                                   image_out_tag="contrast_fake")

                    fake_planet.connect_database(self._m_data_base)
                    fake_planet.run()

                    if self.m_pca_module == "PSFSubtractionModule":

                        psf_sub = PSFSubtractionModule(name_in="pca_contrast",
                                                       pca_number=self.m_pca_number,
                                                       images_in_tag="contrast_fake",
                                                       reference_in_tag="contrast_fake",
                                                       res_arr_out_tag="contrast_res_arr_out",
                                                       res_arr_rot_out_tag="contrast_res_arr_rot_out",
                                                       res_mean_tag="contrast_res_mean",
                                                       res_median_tag="contrast_res_median",
                                                       res_var_tag="contrast_res_var",
                                                       res_rot_mean_clip_tag="contrast_res_rot_mean_clip",
                                                       basis_out_tag="contrast_basis_out",
                                                       image_ave_tag="contrast_image_ave",
                                                       psf_model_tag="contrast_psf_model",
                                                       ref_prep_tag="contrast_ref_prep",
                                                       prep_tag="contrast_prep",
                                                       extra_rot=self.m_extra_rot,
                                                       cent_size=self.m_mask,
                                                       cent_mask_tag="contrast_cent_mask",
                                                       verbose=False)

                    elif self.m_pca_module == "FastPCAModule":

                        if self.m_mask > 0.:
                            warnings.warn("The central mask is not implemented in FastPCAModule.")

                        psf_sub = FastPCAModule(name_in="pca_contrast",
                                                pca_numbers=self.m_pca_number,
                                                images_in_tag="contrast_fake",
                                                reference_in_tag="contrast_fake",
                                                res_mean_tag="contrast_res_mean",
                                                res_median_tag=None,
                                                res_arr_out_tag=None,
                                                res_rot_mean_clip_tag=None,
                                                extra_rot=self.m_extra_rot,
                                                verbose=False)

                    else:
                        raise ValueError("The pca_module should be either PSFSubtractionModule or "
                                         "FastPCAModule.")

                    psf_sub.connect_database(self._m_data_base)
                    psf_sub.run()

                    res_input_port = self.add_input_port("contrast_res_mean")
                    im_res = res_input_port.get_all()

                    if len(im_res.shape) == 3:
                        if im_res.shape[0] == 1:
                            im_res = np.squeeze(im_res, axis=0)
                        else:
                            raise ValueError("Multiple residual images found, expecting only one.")

                    if count == 1 and iteration == 1:
                        self.m_pca_out_port.set_all(im_res, data_dim=3)
                    else:
                        self.m_pca_out_port.append(im_res, data_dim=3)

                    list_fpf.append(self._false_alarm(im_res, x_fake, y_fake, self.m_aperture))

                    if abs(fpf_threshold-list_fpf[-1]) < self.m_accuracy*fpf_threshold:
                        if len(list_fpf) > 1:
                            if (fpf_threshold > list_fpf[-2] and fpf_threshold < list_fpf[-1]) or \
                               (fpf_threshold < list_fpf[-2] and fpf_threshold > list_fpf[-1]):

                                fpf_interp = interp1d(list_fpf[-2:], list_mag[-2:], 'linear')
                                fake_mag[m, n] = fpf_interp(fpf_threshold)

                                print
                                break

                    if list_fpf[-1] < fpf_threshold:
                        if list_mag[-1]+mag_step in list_mag:
                            mag_step /= 2.

                        list_mag.append(list_mag[-1]+mag_step)

                    else:
                        if np.size(list_fpf) > 2 and \
                           list_mag[-1] < list_mag[-2] and list_mag[-2] < list_mag[-3] and \
                           list_fpf[-1] > list_fpf[-2] and list_fpf[-2] < list_fpf[-3]:

                            warnings.warn("Magnitude decreases but false positive fraction "
                                          "increases. Adjusting magnitude to 7.5 and step "
                                          "size to 0.1 ...")

                            list_mag[-1] = 7.5
                            mag_step = 0.1

                        else:
                            if list_mag[-1]-mag_step in list_mag:
                                mag_step /= 2.

                            list_mag.append(list_mag[-1]-mag_step)

                    if list_mag[-1] <= 0.:
                        warnings.warn("The relative magnitude has become smaller or equal to "
                                      "zero. Adjusting magnitude to 7.5 and step size to 0.1 ...")

                        list_mag[-1] = 7.5
                        mag_step = 0.1

                    iteration += 1

                    if iteration == 50:
                        fake_mag[m, n] = np.nan

                        print
                        break

                count += 1

        contrast = np.transpose(np.column_stack((pos_r*pixscale,
                                                 np.nanmean(fake_mag, axis=1),
                                                 np.nanvar(fake_mag, axis=1),
                                                 fake_fpf)))

        self.m_contrast_out_port.set_all(contrast, data_dim=2)

        self.m_pca_out_port.add_history_information("Contrast limits",
                                                    str(self.m_sigma)+" sigma")

        self.m_contrast_out_port.add_history_information("Contrast limits",
                                                         str(self.m_sigma)+" sigma")

        self.m_pca_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_contrast_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_pca_out_port.close_port()
