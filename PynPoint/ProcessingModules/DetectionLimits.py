"""
Modules for determining detection limits.
"""

from __future__ import absolute_import

import math
import sys
import warnings

import numpy as np

from scipy.interpolate import interp1d

from PynPoint.Core.Processing import ProcessingModule
from PynPoint.Util.AnalysisTools import false_alarm, student_fpf, fake_planet
from PynPoint.Util.ImageTools import create_mask
from PynPoint.Util.PSFSubtractionTools import pca_psf_subtraction
from PynPoint.Util.Residuals import combine_residuals


class ContrastCurveModule(ProcessingModule):
    """
    Module to calculate contrast limits by iterating towards a threshold for the false positive
    fraction, with a correction for small sample statistics.
    """

    def __init__(self,
                 name_in="contrast",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 pca_out_tag=None,
                 contrast_out_tag="contrast_limits",
                 separation=(0.1, 1., 0.01),
                 angle=(0., 360., 60.),
                 magnitude=(7.5, 1.),
                 threshold=("sigma", 5.),
                 accuracy=1e-1,
                 psf_scaling=1.,
                 aperture=0.05,
                 ignore=False,
                 pca_number=20,
                 norm=False,
                 cent_size=None,
                 edge_size=None,
                 extra_rot=0.,
                 **kwargs):
        """
        Constructor of ContrastCurveModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that contains the stack with images.
        :type image_in_tag: str
        :param psf_in_tag: Tag of the database entry that contains the reference PSF that is used
                           as fake planet. Can be either a single image (2D) or a cube (3D) with
                           the dimensions equal to *image_in_tag*.
        :type psf_in_tag: str
        :param pca_out_tag: Tag of the database entry that contains all the residuals of the PSF
                            subtraction for each position and optimization step. No data is written
                            if set to None.
        :type pca_out_tag: str
        :param contrast_out_tag: Tag of the database entry that contains the separation,
                                 azimuthally averaged contrast limits, the azimuthal variance of
                                 the contrast limits, and the threshold of the false positive
                                 fraction associated with sigma.
        :type contrast_out_tag: str
        :param separation: Range of separations (arcsec) where the contrast is calculated. Should
                           be specified as (lower limit, upper limit, step size). Apertures that
                           fall within the mask radius or beyond the image size are removed.
        :type separation: (float, float, float)
        :param angle: Range of position angles (deg) where the contrast is calculated. Should be
                      specified as (lower limit, upper limit, step size), measured counterclockwise
                      with respect to the vertical image axis, i.e. East of North.
        :type angle: (float, float, float)
        :param magnitude: Initial magnitude value and step size for the fake planet, specified
                          as (planet magnitude, magnitude step size).
        :type magnitude: (float, float)
        :param threshold: Detection threshold for the contrast curve, either in terms of "sigma"
                          or the false positive fraction (FPF). The value is a tuple, for example
                          provided as ("sigma", 5.) or ("fpf", 1e-6). Note that when sigma is fixed,
                          the false positive fraction will change with separation. Also, sigma only
                          corresponds to the standard deviation of a normal distribution at large
                          separations (i.e., large number of samples).
        :type threshold: tuple(str, float)
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
        :param ignore: Ignore the two neighboring apertures that may contain self-subtraction from
                       the planet.
        :type ignore: bool
        :param pca_number: Number of principal components used for the PSF subtraction.
        :type pca_number: int
        :param norm: Normalization of each image by its Frobenius norm.
        :type norm: bool
        :param cent_size: Central mask radius (arcsec). No mask is used when set to None.
        :type cent_size: float
        :param edge_size: Outer edge radius (arcsec) beyond which pixels are masked. No outer mask
                          is used when set to None. If the value is larger than half the image size
                          then it will be set to half the image size.
        :type edge_size: float
        :param extra_rot: Additional rotation angle of the images in clockwise direction (deg).
        :type extra_rot: float
        :param \**kwargs:
            See below.

        :Keyword arguments:
            **sigma** (*float*) -- Detection threshold in units of sigma. Note that as sigma is
            fixed, the confidence level (and false positive fraction) change with separation. This
            parameter will be deprecated. Please use the *threshold* parameter instead.

        :return: None
        """

        if "sigma" in kwargs:
            self.m_theshold = ("sigma", kwargs["sigma"])

            warnings.warn("The 'sigma' parameter will be deprecated in a future release. It is "
                          "recommended to use the 'threshold' parameter instead.",
                          DeprecationWarning)

        super(ContrastCurveModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        if pca_out_tag is None:
            self.m_pca_out_port = None
        else:
            self.m_pca_out_port = self.add_output_port(pca_out_tag)

        self.m_contrast_out_port = self.add_output_port(contrast_out_tag)

        self.m_image_in_tag = image_in_tag
        self.m_psf_in_tag = psf_in_tag

        self.m_separation = separation
        self.m_angle = angle
        self.m_accuracy = accuracy
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_threshold = threshold
        self.m_aperture = aperture
        self.m_ignore = ignore
        self.m_pca_number = pca_number
        self.m_norm = norm
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_extra_rot = extra_rot

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

        parang = self.m_image_in_port.get_attribute("PARANG")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale

        if self.m_edge_size is not None:
            self.m_edge_size /= pixscale

        self.m_aperture /= pixscale

        if psf.ndim == 3 and psf.shape[0] != images.shape[0]:
            raise ValueError('The number of frames in psf_in_tag does not match with the number of '
                             'frames in image_in_tag. The DerotateAndStackModule can be used to '
                             'average the PSF frames (without derotating) before applying the '
                             'ContrastCurveModule.')

        center = np.array([images.shape[2]/2., images.shape[1]/2.])

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

        fake_mag = np.zeros((len(pos_r), len(pos_t)))
        fake_fpf = np.zeros((len(pos_r)))

        count = 1

        sys.stdout.write("Running ContrastCurveModule...\n")
        sys.stdout.flush()

        for i, sep in enumerate(pos_r):

            if self.m_threshold[0] == "sigma":
                fpf_threshold = student_fpf(sigma=self.m_threshold[1],
                                            radius=sep,
                                            size=self.m_aperture,
                                            ignore=self.m_ignore)

            elif self.m_threshold[0] == "fpf":
                fpf_threshold = self.m_threshold[1]

            else:
                raise ValueError("Threshold type not recognized.")

            fake_fpf[i] = fpf_threshold

            for j, ang in enumerate(pos_t):
                sys.stdout.write("Processing position " + str(count) + " out of " + \
                      str(np.size(fake_mag)))
                sys.stdout.flush()

                x_fake = center[0] + sep*math.cos(np.radians(ang+90.-self.m_extra_rot))
                y_fake = center[1] + sep*math.sin(np.radians(ang+90.-self.m_extra_rot))

                num_mag = np.size(fake_mag[i, 0:j])
                num_nan = np.size(np.where(np.isnan(fake_mag[i, 0:j])))

                if j == 0 or num_mag-num_nan == 0:
                    list_mag = [self.m_magnitude[0]]
                    mag_step = self.m_magnitude[1]

                else:
                    list_mag = [np.nanmean(fake_mag[i, 0:j])]
                    mag_step = 0.1

                list_fpf = []

                iteration = 1

                while True:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                    mag = list_mag[-1]

                    fake = fake_planet(self.m_image_in_port.get_all(), psf,
                                       parang, (sep, ang),
                                       mag, self.m_psf_scaling,
                                       interpolation="spline")

                    im_shape = (fake.shape[-2], fake.shape[-1])
                    mask = create_mask(im_shape, [self.m_cent_size, self.m_edge_size])

                    _, im_res = pca_psf_subtraction(images=fake*mask,
                                                    angles=-1.*parang+self.m_extra_rot,
                                                    pca_number=self.m_pca_number)

                    stack = combine_residuals(method="mean", res_rot=im_res)

                    if self.m_pca_out_port is not None:
                        if count == 1 and iteration == 1:
                            self.m_pca_out_port.set_all(stack, data_dim=3)
                        else:
                            self.m_pca_out_port.append(stack, data_dim=3)

                    _, _, fpf = false_alarm(image=stack,
                                            x_pos=x_fake,
                                            y_pos=y_fake,
                                            size=self.m_aperture,
                                            ignore=self.m_ignore)

                    list_fpf.append(fpf)

                    if abs(fpf_threshold-list_fpf[-1]) < self.m_accuracy*fpf_threshold:
                        if len(list_fpf) == 1:
                            fake_mag[i, j] = list_mag[0]

                            sys.stdout.write("\n")
                            sys.stdout.flush()

                            break

                        else:
                            if (fpf_threshold > list_fpf[-2] and fpf_threshold < list_fpf[-1]) or \
                               (fpf_threshold < list_fpf[-2] and fpf_threshold > list_fpf[-1]):

                                fpf_interp = interp1d(list_fpf[-2:], list_mag[-2:], 'linear')
                                fake_mag[i, j] = fpf_interp(fpf_threshold)

                                sys.stdout.write("\n")
                                sys.stdout.flush()

                                break

                            else:
                                pass

                    if list_fpf[-1] < fpf_threshold:
                        if list_mag[-1]+mag_step in list_mag:
                            mag_step /= 2.

                        list_mag.append(list_mag[-1]+mag_step)

                    else:
                        if np.size(list_fpf) > 2 and \
                           list_mag[-1] < list_mag[-2] and list_mag[-2] < list_mag[-3] and \
                           list_fpf[-1] > list_fpf[-2] and list_fpf[-2] < list_fpf[-3]:

                            warnings.warn("Magnitude decreases but false positive fraction "
                                          "increases. Adjusting magnitude to %s and step size "
                                          "to %s" % (list_mag[-3], mag_step/2.))

                            list_fpf = []
                            list_mag = [list_mag[-3]]
                            mag_step /= 2.

                        else:
                            if list_mag[-1]-mag_step in list_mag:
                                mag_step /= 2.

                            list_mag.append(list_mag[-1]-mag_step)

                    if list_mag[-1] <= 0.:
                        warnings.warn("The relative magnitude has become smaller or equal to "
                                      "zero. Adjusting magnitude to 7.5 and step size to 0.1.")

                        list_mag[-1] = 7.5
                        mag_step = 0.1

                    iteration += 1

                    if iteration == 50:
                        warnings.warn("ContrastModule could not converge at the position of "
                                      "%s arcsec and %s deg." % (sep*pixscale, ang))

                        fake_mag[i, j] = np.nan

                        sys.stdout.write("\n")
                        sys.stdout.flush()

                        break

                count += 1

        result = np.column_stack((pos_r*pixscale,
                                  np.nanmean(fake_mag, axis=1),
                                  np.nanvar(fake_mag, axis=1),
                                  fake_fpf))

        self.m_contrast_out_port.set_all(result, data_dim=2)

        sys.stdout.write("Running ContrastCurveModule... [DONE]\n")
        sys.stdout.flush()

        history = str(self.m_threshold[0])+" = "+str(self.m_threshold[1])

        if self.m_pca_out_port is not None:
            self.m_pca_out_port.add_history_information("ContrastCurveModule", history)
            self.m_pca_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_contrast_out_port.add_history_information("ContrastCurveModule", history)
        self.m_contrast_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_contrast_out_port.close_port()
