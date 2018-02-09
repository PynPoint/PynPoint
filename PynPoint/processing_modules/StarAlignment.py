"""
Modules for locating and aligning of the star.
"""

import math
import sys

import numpy as np
import cv2

from skimage.feature import register_translation
from skimage.transform import rescale
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from scipy.optimize import curve_fit

from PynPoint.util.Progress import progress
from PynPoint.core.Processing import ProcessingModule


class StarExtractionModule(ProcessingModule):
    """
    Module to locate the position of the star in each image.
    """

    def __init__(self,
                 name_in="star_cutting",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_cut",
                 image_size=2.,
                 fwhm_star=0.2):
        """
        Constructor of StarExtractionModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry with the images that are written as
                              output. Should be different from *image_in_tag*.
        :type image_out_tag: str
        :param image_size: Cropped size (arcsec) of the images.
        :type image_size: float
        :param fwhm_star: Full width at half maximum (arcsec) of the Gaussian kernel that is used
                          to convolve the images.
        :type fwhm_star: float

        :return: None
        """

        super(StarExtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_inout_port = self.add_output_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_image_size = image_size
        self.m_fwhm_star = fwhm_star # 7 pix / 0.2 arcsec is good for L-band data
        self.count = 0

    def run(self):
        """
        Run method of the module. Locates the position of the star (only pixel precision) by
        selecting the largest pixel value. A Gaussian kernel with a FWHM similar to the PSF is
        used to smooth away the contribution of bad pixels which may have higher values than the
        peak of the PSF. Images are cropped and written to an output port. The position of the
        star is attached as a non-static attribute (STAR_POSITION) to the database tag with the
        input images.

        :return: None
        """

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        psf_radius = int((self.m_image_size / 2.0) / pixscale)

        self.m_fwhm_star /= pixscale
        self.m_fwhm_star = int(self.m_fwhm_star)

        star_positions = []

        def cut_psf(current_image):

            sigma = self.m_fwhm_star/math.sqrt(8.*math.log(2.))

            kernel_size = (self.m_fwhm_star*2 + 1, self.m_fwhm_star*2 + 1)

            search_image = cv2.GaussianBlur(current_image,
                                            kernel_size,
                                            sigma)

            argmax = np.unravel_index(search_image.argmax(), search_image.shape)

            if argmax[0] <= psf_radius or argmax[1] <= psf_radius \
                    or argmax[0] + psf_radius > current_image.shape[0] \
                    or argmax[1] + psf_radius > current_image.shape[1]:

                raise ValueError('Highest value is near the border. PSF size is too '
                                 'large to be cut (image index = '+str(self.count)+').')

            cut_image = current_image[int(argmax[0] - psf_radius):int(argmax[0] + psf_radius),
                                      int(argmax[1] - psf_radius):int(argmax[1] + psf_radius)]

            star_positions.append(argmax)

            self.count += 1

            return cut_image

        self.apply_function_to_images(cut_psf,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarExtractionModule...",
                                      num_images_in_memory=memory)

        self.m_image_inout_port.add_attribute("STAR_POSITION",
                                              np.asarray(star_positions),
                                              static=False)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_history_information("Star extract",
                                                      "Maximum in smoothed image")

        self.m_image_out_port.close_port()


class StarAlignmentModule(ProcessingModule):
    """
    Module to align the images with a cross-correlation in Fourier space.
    """

    def __init__(self,
                 name_in="star_align",
                 image_in_tag="im_arr",
                 ref_image_in_tag=None,
                 image_out_tag="im_arr_aligned",
                 interpolation="fft",
                 accuracy=10,
                 resize=1,
                 num_references=10):
        """
        Constructor of StarAlignmentModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with the stack of images that is read as
                             input.
        :type image_in_tag: str
        :param ref_image_in_tag: Tag of the database entry with the reference image(s) that are
                                 read as input.
        :type ref_image_in_tag: str
        :param image_out_tag: Tag of the database entry with the images that are written as
                              output.
        :type image_out_tag: str
        :param interpolation: Type of interpolation that is used for shifting the images (spline,
                              bilinear, or fft).
        :type interpolation: str
        :param accuracy: Upsampling factor for the cross-correlation. Images will be registered
                         to within 1/accuracy of a pixel.
        :type accuracy: float
        :param resize: Scaling factor for the up/down-sampling before the images are shifted.
        :type resize: float
        :param num_references: Number of reference images for the cross-correlation.
        :type num_references: int

        :return: None
        """

        super(StarAlignmentModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if ref_image_in_tag is not None:
            self.m_ref_image_in_port = self.add_input_port(ref_image_in_tag)
        else:
            self.m_ref_image_in_port = None

        self.m_interpolation = interpolation
        self.m_accuracy = accuracy
        self.m_resize = resize
        self.m_num_references = num_references

    def run(self):
        """
        Run method of the module. Applies a cross-correlation of the input images with respect to
        a stack of reference images, rescales the image dimensions, and shifts the images to a
        common center.

        :return: None
        """

        memory = self._m_config_port.get_attribute("MEMORY")

        if self.m_ref_image_in_port is not None:
            if len(self.m_ref_image_in_port.get_shape()) == 3:
                ref_images = np.asarray(self.m_ref_image_in_port.get_all())
            elif len(self.m_ref_image_in_port.get_shape()) == 2:
                ref_images = np.array([self.m_ref_image_in_port.get_all(),])
            else:
                raise ValueError("reference Image needs to be 2 D or 3 D.")
        else:
            ref_images = self.m_image_in_port[np.sort(
                np.random.choice(self.m_image_in_port.get_shape()[0],
                                 self.m_num_references,
                                 replace=False)), :, :]

        def align_image(image_in):

            offset = np.array([0.0, 0.0])
            for i in range(self.m_num_references):
                norm = max(np.amax(np.abs(ref_images[i, ])), np.amax(np.abs(image_in[i, ])))

                tmp_offset, _, _ = register_translation(ref_images[i, ]/norm,
                                                        image_in/norm,
                                                        upsample_factor=self.m_accuracy)
                offset += tmp_offset

            offset /= float(self.m_num_references)
            offset *= self.m_resize

            if self.m_resize is not 1:
                sum_before = np.sum(image_in)
                tmp_image = rescale(image=np.asarray(image_in),
                                    scale=(self.m_resize,
                                           self.m_resize),
                                    order=5,
                                    mode="reflect")
                sum_after = np.sum(tmp_image)

                # Conserve flux because the rescale function normalizes all values to [0:1].
                tmp_image = tmp_image*(sum_before/sum_after)

            else:
                tmp_image = image_in

            if self.m_interpolation == "fft":
                tmp_image_spec = fourier_shift(np.fft.fftn(tmp_image), offset)
                tmp_image = np.fft.ifftn(tmp_image_spec).real

            elif self.m_interpolation == "spline":
                tmp_image = shift(tmp_image, offset, order=5)

            elif self.m_interpolation == "bilinear":
                tmp_image = shift(tmp_image, offset, order=1)

            else:
                raise ValueError("Interpolation needs to be spline, bilinear or fft")

            return tmp_image

        self.apply_function_to_images(align_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarAlignmentModule...",
                                      num_images_in_memory=memory)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        tmp_pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        tmp_pixscale /= self.m_resize
        self.m_image_out_port.add_attribute("PIXSCALE", tmp_pixscale)

        history = "cross-correlation with up-sampling factor " + str(self.m_accuracy)
        self.m_image_out_port.add_history_information("PSF alignment",
                                                      history)
        self.m_image_out_port.close_port()


class LocateStarModule(ProcessingModule):
    """
    Module for locating the position of the star.
    """

    def __init__(self,
                 name_in="locate_star",
                 data_tag="im_arr",
                 gaussian_fwhm=0.2):
        """
        Constructor of LocateStarModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param data_tag: Tag of the database entry for which the star positions are written as
                         attributes.
        :type data_tag: str
        :param gaussian_fwhm: Full width at half maximum (arcsec) of the Gaussian kernel that is
                              used to smooth the image before the star is located.
        :type gaussian_fwhm: float
        :return: None
        """

        super(LocateStarModule, self).__init__(name_in)

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

        self.m_gaussian_fwhm = gaussian_fwhm

    def run(self):
        """
        Run method of the module. Smooths the image with a Gaussian kernel, finds the largest
        pixel value, and writes the STAR_POSITION attribute.

        :return: None
        """

        pixscale = self.m_data_in_port.get_attribute("PIXSCALE")
        self.m_gaussian_fwhm /= pixscale

        sigma = self.m_gaussian_fwhm/math.sqrt(8.*math.log(2.))

        star_position = np.zeros((self.m_data_in_port.get_shape()[0], 2), dtype=np.int64)

        for i in range(self.m_data_in_port.get_shape()[0]):
            progress(i, self.m_data_in_port.get_shape()[0], "Running LocateStarModule...")

            im_smooth = gaussian_filter(self.m_data_in_port[i],
                                        sigma,
                                        truncate=4.)

            star_position[i, :] = np.unravel_index(im_smooth.argmax(), im_smooth.shape)

        sys.stdout.write("Running LocateStarModule... [DONE]\n")
        sys.stdout.flush()

        self.m_data_out_port.add_attribute("STAR_POSITION",
                                           star_position,
                                           static=False)

        self.m_data_out_port.close_port()


class StarCenteringModule(ProcessingModule):
    """
    Module for centering the star by fitting a 2D Gaussian profile.
    """

    def __init__(self,
                 name_in="centering",
                 image_in_tag="im_arr",
                 image_out_tag="im_center",
                 fit_out_tag="center_fit",
                 method="full",
                 interpolation="spline",
                 **kwargs):
        """
        Constructor of StarCenteringModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry with the centered images that are written
                              as output. Should be different from *image_in_tag*.
        :type image_out_tag: str
        :param fit_out_tag: Tag of the database entry with the best-fit results of the 2D Gaussian
                            fit and the 1sigma errors. Data is written in the following format:
                            x offset (arcsec), x offset error (arcsec), y offset (arcsec), y offset
                            error (arcsec), FWHM major axis (arcsec), FWHM major axis error
                            (arcsec), FWHM minor axis (arcsec), FWHM minor axis error
                            (arcsec), amplitude (counts), amplitude error (counts), angle (deg)
                            measured in counterclockwise direction with respect to the upward
                            direction (i.e., East of North).
        :type fit_out_tag: str
        :param method: Fit and shift all the images individually ("full") or only fit the mean of
                       the cube and shift all images to that location ("mean"). The "mean" method
                       could be used after running the StarAlignmentModule.
        :type method: str
        :param interpolation: Type of interpolation that is used for shifting the images (fft,
                              spline, or bilinear).
        :type interpolation: str

        :return: None
        """

        if "guess" in kwargs:
            self.m_guess = kwargs["guess"]
        else:
            self.m_guess = (0., 0., 1., 1., 1., 0.)

        super(StarCenteringModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_fit_out_port = self.add_output_port(fit_out_tag)

        self.m_method = method
        self.m_interpolation = interpolation

    def run(self):
        """
        Run method of the module. Uses a non-linear least squares (Levenberg-Marquardt) to fit the
        the individual images or the mean of the stack with a 2D Gaussian profile, shifts the
        images with subpixel precision, and writes the centered images and the fitting results.

        :return: None
        """

        def _2d_gaussian((x_grid, y_grid), x_center, y_center, fwhm_x, fwhm_y, amp, theta):
            xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

            x_diff = xx_grid - x_center
            y_diff = yy_grid - y_center

            sigma_x = fwhm_x/math.sqrt(8.*math.log(2.))
            sigma_y = fwhm_y/math.sqrt(8.*math.log(2.))

            a_gauss = 0.5 * ((np.cos(theta)/sigma_x)**2 + (np.sin(theta)/sigma_y)**2)
            b_gauss = 0.5 * ((np.sin(2.*theta)/sigma_x**2) - (np.sin(2.*theta)/sigma_y**2))
            c_gauss = 0.5 * ((np.sin(theta)/sigma_x)**2 + (np.cos(theta)/sigma_y)**2)

            gaussian = amp*np.exp(-(a_gauss*x_diff**2 + b_gauss*x_diff*y_diff + c_gauss*y_diff**2))

            return np.ravel(gaussian)

        def _least_squares(image):
            im_ravel = np.ravel(image)

            popt, pcov = curve_fit(_2d_gaussian,
                                   (self.m_x_grid, self.m_y_grid),
                                   im_ravel,
                                   p0=self.m_guess,
                                   sigma=None,
                                   method='lm')

            perr = np.sqrt(np.diag(pcov))

            res = np.asarray((popt[0]*pixscale, perr[0]*pixscale,
                              popt[1]*pixscale, perr[1]*pixscale,
                              popt[2]*pixscale, perr[2]*pixscale,
                              popt[3]*pixscale, perr[3]*pixscale,
                              popt[4], perr[4],
                              math.degrees(popt[5])%360., math.degrees(perr[5])))

            self.m_fit_out_port.append(res, data_dim=2)

            return popt

        def _centering(image):

            if self.m_method == "full":
                popt = _least_squares(image)

            elif self.m_method == "mean":
                popt = self.m_popt

            if self.m_interpolation == "fft":
                fft_shift = fourier_shift(np.fft.fftn(image), (-popt[1], -popt[0]))
                im_center = np.fft.ifftn(fft_shift).real

            elif self.m_interpolation == "spline":
                im_center = shift(image, (-popt[1], -popt[0]), order=5)

            elif self.m_interpolation == "bilinear":
                im_center = shift(image, (-popt[1], -popt[0]), order=1)

            return im_center

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        self.m_fit_out_port.del_all_data()
        self.m_fit_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        nimages = self.m_image_in_port.get_shape()[0]
        nstacks = int(float(nimages)/float(memory))
        npix = self.m_image_in_port.get_shape()[1]

        if npix%2 == 0:
            self.m_x_grid = self.m_y_grid = np.linspace(-npix/2+0.5, npix/2-0.5, npix)
        elif npix%2 == 1:
            self.m_x_grid = self.m_y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)

        if self.m_method == "mean":
            im_mean = np.zeros((npix, npix))

            for i in range(nstacks):
                im_mean += np.sum(self.m_image_in_port[i*memory:i*memory+memory, ],
                                  axis=0)

            if nimages%memory > 0:
                im_mean += np.sum(self.m_image_in_port[nstacks*memory:nimages, ],
                                  axis=0)

            im_mean /= float(nimages)

            self.m_popt = _least_squares(im_mean)

        self.apply_function_to_images(_centering,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarCenteringModule...",
                                      num_images_in_memory=memory)

        self.m_image_out_port.add_history_information("Centering",
                                                      "2D Gaussian fit")
        self.m_fit_out_port.add_history_information("Centering",
                                                    "2D Gaussian fit")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_fit_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()
