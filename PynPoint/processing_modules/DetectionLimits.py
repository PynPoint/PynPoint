"""
Modules for calculating detection limits.
"""

import math
import warnings

import numpy as np
from photutils import aperture_photometry, CircularAperture
from scipy.interpolate import interp1d
from scipy.stats import t, norm
from scipy.ndimage import shift

from PynPoint.core import ProcessingModule
from PynPoint.processing_modules import PSFSubtractionModule


class ContrastModule(ProcessingModule):
    """
    Module to calculate contrast limits.
    """

    def __init__(self,
                 name_in="contrast",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 pca_out_tag="im_fake",
                 contrast_out_tag="contrast",
                 separation=(0.1, 1., 0.01),
                 angle=(0., 360., 10.),
                 sigma=5.,
                 accuracy=1e-1,
                 magnitude=(10., 2.),
                 scaling=1.,
                 aperture=0.05,
                 pca_number=20,
                 mask=0.02):
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
        :param pca_out_tag: Tag of the database entry that contains all the output of the PSF
                            subtraction with the optimization of the fake planet brightness.
        :type pca_out_tag: str
        :param contrast_out_tag: Tag of the database entry that contains the azimuthally averaged
                                 contrast as well as the azimuthal standard deviation of the
                                 contrast.
        :type contrast_out_tag: str
        :param separation: Separations (arcsec) where the contrast is calculated. Should be
                           specified as (lower limit, upper limit, step size). Apertures that
                           fall within the mask radius or beyond the image size are removed.
        :type separation: tuple
        :param angle: Position angles (deg) where the contrast is calculated. Should be specified
                      as (lower limit, upper limit, step size), measured counterclockwise with
                      respect to the vertical image axis, i.e. East of North.
        :type angle: tuple
        :param sigma: Confidence level, in units of sigma, at which the contrast is calculated.
        :type sigma: float
        :param accuracy: Accuracy relative to the False Probability Fraction threshold. When the
                         accuracy condition is met, the final magnitude is calculated with a
                         linear interpolation.
        :type accuracy: float
        :param magnitude: Initial magnitude value and step size for the fake planet, specified
                          as (planet magnitude, magnitude step size).
        :type magnitude: tuple
        :param scaling: Scaling factor for the brightness of the fake planet signal.
        :type scaling: float
        :param aperture: Aperture radius (arcsec) for the calculation of the False Positive
                         Fraction.
        :type aperture: float
        :param pca_number: Number of principle components used for the PSF subtraction
        :type pca_number: int
        :param mask: Mask radius (arcsec) for the PSF subtraction
        :type mask: float
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
        self.m_fake_data_port = self.add_output_port("contrast_fake")

        self.m_separation = separation
        self.m_angle = angle
        self.m_sigma = sigma
        self.m_accuracy = accuracy
        self.m_magnitude = magnitude
        self.m_scaling = scaling
        self.m_pca_number = pca_number
        self.m_mask = mask
        self.m_aperture = aperture

        self._m_psf_sub = PSFSubtractionModule(name_in="pca_contrast",
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
                                               extra_rot=0.,
                                               cent_size=self.m_mask,
                                               cent_mask_tag="contrast_cent_mask",
                                               verbose=False)

    def connect_database(self,
                         data_base_in):

        self._m_psf_sub.connect_database(data_base_in)
        super(ContrastModule, self).connect_database(data_base_in)

    @staticmethod
    def _false_alarm(image,
                     x_pos,
                     y_pos,
                     size):

        center = (np.size(image, 0)/2., np.size(image, 1)/2.)
        radius = math.sqrt((center[0]-y_pos)**2.+(center[1]-x_pos)**2.)
        num_ap = int(2.*math.pi*radius/size)

        theta_step = 2.*math.pi/float(num_ap)

        ap_phot = np.zeros(num_ap)
        ap_theta = np.arange(0, 2.*math.pi, theta_step)

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

        fpf = 1. - t.cdf(t_test, num_ap-2)

        return fpf

    def run(self):
        """
        Run method of the module.

        :return: None
        """

        images = self.m_image_in_port.get_all()
        psf = self.m_psf_in_port.get_all()

        # TODO update for global config
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_aperture /= pixscale*1e-3
        self.m_mask /= pixscale*1e-3*images.shape[1]/2.

        fpf_threshold = 1. - norm(0., 1.).cdf(self.m_sigma)

        if psf.ndim == 3 and psf.shape[0] != images.shape[0]:
            warnings.warn('The number of frames in psf_in_tag does not match with the number of '
                          'frames in image_in_tag. Using the mean of psf_in_tag as fake planet.')

        parang = self.m_image_in_port.get_attribute("NEW_PARA")
        center = np.array([images.shape[2]/2., images.shape[1]/2.])

        pos_r = np.arange(self.m_separation[0]/(pixscale*1e-3),
                          self.m_separation[1]/(pixscale*1e-3),
                          self.m_separation[2]/(pixscale*1e-3))

        pos_t = np.arange(self.m_angle[0]-90.,
                          self.m_angle[1]-90.,
                          self.m_angle[2])

        index_del = np.argwhere(pos_r-self.m_aperture < self.m_mask*images.shape[1]/2.)
        pos_r = np.delete(pos_r, index_del)

        index_del = np.argwhere(pos_r+self.m_aperture > images.shape[1]/2.)
        pos_r = np.delete(pos_r, index_del)

        fake_mag = np.zeros((len(pos_r), len(pos_t)))

        count = 1

        # TODO make parallel?
        for m, radial in enumerate(pos_r):
            for n, theta in enumerate(pos_t):
                print "Processing fake planet " + str(count) + " out of " + \
                      str(np.size(fake_mag)) + "..."

                x_fake = center[0] + radial*math.cos(theta*math.pi/180.)
                y_fake = center[1] + radial*math.sin(theta*math.pi/180.)

                mag_step = self.m_magnitude[1]

                list_mag = [self.m_magnitude[0]]
                list_fpf = []

                while True:
                    im_tmp = np.copy(images)
                    psf_tmp = np.copy(psf)

                    flux_ratio = 10**(-list_mag[-1]/2.5)

                    for i in range(np.size(im_tmp, 0)):
                        x_shift = radial*math.cos((theta-parang[i])*math.pi/180.)
                        y_shift = radial*math.sin((theta-parang[i])*math.pi/180.)

                        if psf_tmp.ndim == 2:
                            psf_tmp = shift(psf_tmp, (y_shift, x_shift), order=5, mode='reflect')
                            im_tmp[i,] += self.m_scaling*flux_ratio*psf_tmp

                        elif psf_tmp.ndim == 3:
                            if psf_tmp.shape[0] == im_tmp.shape[0]:
                                psf_tmp[i] = shift(psf_tmp[i,],
                                                   (y_shift, x_shift),
                                                   order=5,
                                                   mode='reflect')

                                im_tmp[i,] += self.m_scaling*flux_ratio*psf_tmp[i,]

                            else:
                                psf_tmp = np.mean(psf_tmp, axis=0)
                                psf_tmp = shift(psf_tmp,
                                                (y_shift, x_shift),
                                                order=5,
                                                mode='reflect')
                                im_tmp[i,] += self.m_scaling*flux_ratio*psf_tmp

                    self.m_fake_data_port.set_all(im_tmp)
                    self.m_fake_data_port.copy_attributes_from_input_port(self.m_image_in_port)

                    self._m_psf_sub.run()

                    self.m_pca_in_port = self.add_input_port("contrast_res_mean")
                    im_pca = self.m_pca_in_port.get_all()

                    if m == 0 and n == 0 and np.size(list_fpf) == 0:
                        self.m_pca_out_port.set_all(im_pca, data_dim=3)
                    else:
                        self.m_pca_out_port.append(im_pca, data_dim=3)

                    list_fpf.append(self._false_alarm(im_pca, x_fake, y_fake, self.m_aperture))

                    # TODO Adjust stepsize for else case?
                    if abs(fpf_threshold-list_fpf[-1]) < self.m_accuracy*fpf_threshold:
                        if (fpf_threshold > list_fpf[-2] and fpf_threshold < list_fpf[-1]) or \
                           (fpf_threshold < list_fpf[-2] and fpf_threshold > list_fpf[-1]):

                            fpf_interp = interp1d(list_fpf[-2:], list_mag[-2:], 'linear')
                            fake_mag[m, n] = fpf_interp(fpf_threshold)

                            break

                    if list_fpf[-1] < fpf_threshold:
                        if list_mag[-1]+mag_step in list_mag:
                            mag_step /= 2.
                        list_mag.append(list_mag[-1]+mag_step)

                    else:
                        if np.size(list_fpf) > 2 and \
                           list_mag[-1] < list_mag[-2] and list_mag[-2] < list_mag[-3] and \
                           list_fpf[-1] > list_fpf[-2] and list_fpf[-2] < list_fpf[-3]:

                            raise ValueError("Magnitude decreases but False Positive Fraction "
                                             "increases. This should not happen, try optimizing "
                                             "the aperture radius.")

                            # list_mag = [list_mag[-3]]
                            # list_fpf = []
                            # mag_step *= 0.1

                        else:
                            if list_mag[-1]-mag_step in list_mag:
                                mag_step /= 2.
                            list_mag.append(list_mag[-1]-mag_step)

                    if list_mag[-1] <= 0.:
                        raise ValueError("The relative magnitude has become smaller or equal to "
                                         "zero. Try changing the aperture and magnitude "
                                         "parameters")

                count += 1

        contrast = np.transpose(np.column_stack((pos_r*pixscale*1e-3,
                                                 np.mean(fake_mag, axis=1),
                                                 np.std(fake_mag, axis=1))))

        # TODO added header to fake output

        self.m_contrast_out_port.set_all(contrast, data_dim=2)

        self.m_contrast_out_port.add_history_information("Contrast false positive fraction",
                                                         fpf_threshold)

        self.m_contrast_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_contrast_out_port.close_port()
