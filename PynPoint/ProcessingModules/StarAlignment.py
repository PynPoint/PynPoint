"""
Modules for locating, aligning, and centering of the star.
"""

import math

import numpy as np
import cv2
import warnings

from skimage.feature import register_translation
from skimage.transform import rescale
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from scipy.optimize import curve_fit

from PynPoint.Core.Processing import ProcessingModule


class StarExtractionModule(ProcessingModule):
    """
    Module to locate the position of the star in each image and to crop all the images around this position.
    """

    def __init__(self,
                 name_in="star_cutting",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_crop",
                 image_size=2.,
                 fwhm_star=0.2,
                 position=None):
        """
        Constructor of StarExtractionModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*. If *image_out_tag* and/or *image_size*
                              is set to None then only the STAR_POSITION attributes will be written
                              to *image_in_tag* and *image_out_tag* is not used.
        :type image_out_tag: str
        :param image_size: Cropped image size (arcsec). If *image_out_tag* and/or *image_size* is
                           set to None then only the STAR_POSITION attributes will be written to
                           *image_in_tag* and *image_out_tag* is not used.
        :type image_size: float
        :param fwhm_star: Full width at half maximum (arcsec) of the Gaussian kernel that is used
                          to convolve the images.
        :type fwhm_star: float
        :param position: Subframe that is selected to search for the star. The tuple can contain a
                         single position in pixels and size as (pos_x, pos_y, size), or the position
                         and size can be defined for each image separately in which case the tuple
                         should be 2D (nframes x 3). Setting *position* to None will use the full
                         image to search for the star. If position=(None, None, size) then the center
                         of the image will be used.
        :type position: tuple, float

        :return: None
        """

        super(StarExtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        if image_out_tag is None:
            self.m_image_out_port = None
        else:
            self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_position_out_port = self.add_output_port(image_in_tag)

        self.m_image_size = image_size
        self.m_fwhm_star = fwhm_star
        self.m_position = position
        self.m_count = 0
        self.m_image_out_tag = image_out_tag

    def run(self):
        """
        Run method of the module. Locates the position of the star (only pixel precision) by
        selecting the highest pixel value. A Gaussian kernel with a FWHM similar to the PSF is
        used to smooth away the contribution of bad pixels which may have higher values than the
        peak of the PSF. Images are cropped and written to an output port. The position of the
        star is attached as a non-static attribute (STAR_POSITION) to the database tag with the
        input images.

        :return: None
        """

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if self.m_position is not None:
            self.m_position = np.asarray(self.m_position)

            if self.m_position.ndim == 2 and \
                    self.m_position.shape[0] != self.m_image_in_port.get_shape()[0]:
                raise ValueError("Either a single 'position' should be specified or an array "
                                 "equal in size to the number of images in 'image_in_tag'.")

            if self.m_position[0] is None and self.m_position[1] is None:
                npix = self.m_image_in_port.get_shape()[1]
                self.m_position[0] = npix/2.
                self.m_position[1] = npix/2.

        if self.m_image_size is not None:
            psf_radius = int((self.m_image_size/2.)/pixscale)

        self.m_fwhm_star /= pixscale
        self.m_fwhm_star = int(self.m_fwhm_star)

        star = []

        def crop_image(image):
            sigma = self.m_fwhm_star/math.sqrt(8.*math.log(2.))
            kernel = (self.m_fwhm_star*2 + 1, self.m_fwhm_star*2 + 1)

            if self.m_position is None:
                subimage = image

            else:
                if self.m_position.ndim == 1:
                    pos_x = self.m_position[0]
                    pos_y = self.m_position[1]
                    width = self.m_position[2]
                
                    if pos_x > self.m_image_in_port.get_shape()[1] or \
                            pos_y > self.m_image_in_port.get_shape()[2]:
                        raise ValueError('The indicated position lays outside the image')

                elif self.m_position.ndim == 2:
                    pos_x = self.m_position[self.m_count, 0]
                    pos_y = self.m_position[self.m_count, 1]
                    width = self.m_position[self.m_count, 2]
                
                if pos_y <= width/2. or pos_x <= width/2. \
                        or pos_y+width/2. >= self.m_image_in_port.get_shape()[2]\
                        or pos_x+width/2. >= self.m_image_in_port.get_shape()[1]:
                    warnings.warn("The region for the star extraction exceeds the image")

                subimage = image[int(pos_y-width/2.):int(pos_y+width/2.),
                                 int(pos_x-width/2.):int(pos_x+width/2.)]

            im_smooth = cv2.GaussianBlur(subimage, kernel, sigma)
            argmax = np.asarray(np.unravel_index(im_smooth.argmax(), im_smooth.shape))

            if self.m_position is not None:
                argmax[0] += pos_y-width/2.
                argmax[1] += pos_x-width/2.

            if self.m_image_size is not None:
                if argmax[0] <= psf_radius or argmax[1] <= psf_radius \
                        or argmax[0] + psf_radius >= image.shape[0] \
                        or argmax[1] + psf_radius >= image.shape[1]:

                    raise ValueError('Highest value is near the border. PSF size is too '
                                     'large to be cropped (image index = '+str(self.m_count)+').')

                im_crop = image[int(argmax[0] - psf_radius):int(argmax[0] + psf_radius),
                                int(argmax[1] - psf_radius):int(argmax[1] + psf_radius)]

            star.append(argmax)
            self.m_count += 1

            if self.m_image_size is not None:
                return im_crop

        self.apply_function_to_images(crop_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarExtractionModule...",
                                      num_images_in_memory=memory)

        self.m_position_out_port.add_attribute("STAR_POSITION", np.asarray(star), static=False)

        if self.m_image_size is not None and self.m_image_out_tag is not None:
            self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
            self.m_image_out_port.add_history_information("Star extract", "maximum")
            self.m_image_out_port.close_database()


class StarAlignmentModule(ProcessingModule):
    """
    Module to align the images with a cross-correlation in Fourier space.
    """

    def __init__(self,
                 name_in="star_align",
                 image_in_tag="im_arr",
                 ref_image_in_tag=None,
                 image_out_tag="im_arr_aligned",
                 interpolation="spline",
                 accuracy=10,
                 resize=None,
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
                tmp_offset, _, _ = register_translation(ref_images[i, :, :],
                                                        image_in,
                                                        upsample_factor=self.m_accuracy)
                offset += tmp_offset

            offset /= float(self.m_num_references)
            if self.m_resize is not None:
                offset *= self.m_resize

            if self.m_resize is not None:
                sum_before = np.sum(image_in)
                tmp_image = rescale(image=np.asarray(image_in, dtype=np.float64),
                                    scale=(self.m_resize, self.m_resize),
                                    order=5,
                                    mode="reflect")
                sum_after = np.sum(tmp_image)

                # Conserve flux because the rescale function normalizes all values to [0:1].
                tmp_image = tmp_image*(sum_before/sum_after)

            else:
                tmp_image = image_in

            if self.m_interpolation == "spline":
                tmp_image = shift(tmp_image, offset, order=5)

            elif self.m_interpolation == "bilinear":
                tmp_image = shift(tmp_image, offset, order=1)

            elif self.m_interpolation == "fft":
                tmp_image_spec = fourier_shift(np.fft.fftn(tmp_image), offset)
                tmp_image = np.fft.ifftn(tmp_image_spec).real

            else:
                raise ValueError("Interpolation should be spline, bilinear, or fft.")

            return tmp_image

        self.apply_function_to_images(align_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarAlignmentModule...",
                                      num_images_in_memory=memory)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        tmp_pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if self.m_resize is not None:
            tmp_pixscale /= self.m_resize
        self.m_image_out_port.add_attribute("PIXSCALE", tmp_pixscale)

        history = "cross-correlation with up-sampling factor " + str(self.m_accuracy)
        self.m_image_out_port.add_history_information("PSF alignment",
                                                      history)
        self.m_image_out_port.close_database()


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
        :param interpolation: Type of interpolation that is used for shifting the images (spline,
                              bilinear, or fft).
        :type interpolation: str
        :param \**kwargs:
            See below.

        :Keyword arguments:
             * **guess** (*tuple*) -- Tuple with the initial parameter values for the least
                                      squares fit: center x (pix), center y (pix), FWHM x (pix),
                                      FWHM y (pix), amplitude (counts), angle (deg).

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
            npix = image.shape[0]

            if npix%2 == 0:
                x_grid = y_grid = np.linspace(-npix/2+0.5, npix/2-0.5, npix)
            elif npix%2 == 1:
                x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)

            im_ravel = np.ravel(image)

            popt, pcov = curve_fit(_2d_gaussian,
                                   (x_grid, y_grid),
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

            if self.m_interpolation == "spline":
                im_center = shift(image, (-popt[1], -popt[0]), order=5)

            elif self.m_interpolation == "bilinear":
                im_center = shift(image, (-popt[1], -popt[0]), order=1)

            elif self.m_interpolation == "fft":
                fft_shift = fourier_shift(np.fft.fftn(image), (-popt[1], -popt[0]))
                im_center = np.fft.ifftn(fft_shift).real

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

        if self.m_method == "mean":
            im_mean = np.zeros((npix, npix))

            for i in range(nstacks):
                im_mean += np.sum(self.m_image_in_port[i*memory:i*memory+memory, ], axis=0)

            if nimages%memory > 0:
                im_mean += np.sum(self.m_image_in_port[nstacks*memory:nimages, ], axis=0)

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

        self.m_image_out_port.close_database()


class ShiftForCenteringModule(ProcessingModule):
    """
    Module for shifting of an image.
    """

    def __init__(self,
                 shift_xy,
                 name_in="shift",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_shifted"):
        """
        Constructor of ShiftForCenteringModule.

        :param shift_xy: Tuple (delta_y, delta_x) with the shift in both directions.
        :type shift_xy: tuple, float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(ShiftForCenteringModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_shift = shift_xy

    def run(self):
        """
        Run method of the module. Shifts an image with a fifth order spline interpolation.

        :return: None
        """

        memory = self._m_config_port.get_attribute("MEMORY")

        def image_shift(image_in):
            return shift(image_in, self.m_shift, order=5)

        self.apply_function_to_images(image_shift,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running ShiftForCenteringModule...",
                                      num_images_in_memory=memory)

        self.m_image_out_port.add_history_information("Shifted", str(self.m_shift))
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_database()
