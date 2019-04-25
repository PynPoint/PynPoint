"""
Pipeline modules for locating, aligning, and centering of the star.
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import math
import warnings

import numpy as np

from skimage.feature import register_translation
from skimage.transform import rescale
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
from six.moves import range

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import memory_frames, progress, locate_star
from pynpoint.util.image import crop_image, shift_image, center_pixel


class StarExtractionModule(ProcessingModule):
    """
    Pipeline module to locate the position of the star in each image and to crop all the images
    around this position.
    """

    def __init__(self,
                 name_in="star_cutting",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_crop",
                 index_out_tag=None,
                 image_size=2.,
                 fwhm_star=0.2,
                 position=None):
        """
        Constructor of StarExtractionModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.
        index_out_tag : str
            List with image indices for which the image size is too large to be cropped around the
            brightest pixel. No data is written if set to None. This tag name can be provided to
            the *frames* parameter in
            :class:`~pynpoint.processing.frameselection.RemoveFramesModule`.
        image_size : float
            Cropped image size (arcsec).
        fwhm_star : float
            Full width at half maximum (arcsec) of the Gaussian kernel that is used to smooth the
            images to lower contributions of bad pixels.
        position : tuple(int, int, float)
            Subframe that is selected to search for the star. The tuple can contain a single
            position (pix) and size (arcsec) as (pos_x, pos_y, size), or the position and size can
            be defined for each image separately in which case the tuple should be 2D
            (nframes x 3). Setting *position* to None will use the full image to search for the
            star. If *position=(None, None, size)* then the center of the image will be used.

        Returns
        -------
        NoneType
            None
        """

        super(StarExtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if index_out_tag is None:
            self.m_index_out_port = None
        else:
            self.m_index_out_port = self.add_output_port(index_out_tag)

        self.m_image_size = image_size
        self.m_fwhm_star = fwhm_star
        self.m_position = position

        self.m_count = 0

    def run(self):
        """
        Run method of the module. Locates the position of the star (only pixel precision) by
        selecting the highest pixel value. A Gaussian kernel with a FWHM similar to the PSF is
        used to lower the contribution of bad pixels which may have higher values than the
        peak of the PSF. Images are cropped and written to an output port. The position of the
        star is attached to the input images as the non-static attribute STAR_POSITION (y, x).

        Returns
        -------
        NoneType
            None
        """

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if self.m_position is not None:
            self.m_position = np.asarray(self.m_position)
            nimages = self.m_image_in_port.get_shape()[0]

            if self.m_position.ndim == 2 and self.m_position.shape[0] != nimages:
                raise ValueError("Either a single 'position' should be specified or an array "
                                 "equal in size to the number of images in 'image_in_tag'.")

        self.m_image_size = int(math.ceil(self.m_image_size/pixscale))
        self.m_fwhm_star = int(math.ceil(self.m_fwhm_star/pixscale))

        star = []
        index = []

        def _crop_around_star(image, position, im_size, fwhm):

            if position is None:
                center = None
                width = None

            else:
                if position.ndim == 1:
                    if position[0] is None and position[1] is None:
                        center = None
                    else:
                        center = (int(position[1]), int(position[0]))

                    width = int(math.ceil(position[2]/pixscale))

                elif position.ndim == 2:
                    center = (int(position[self.m_count, 1]), int(position[self.m_count, 0]))
                    width = int(math.ceil(position[self.m_count, 2]/pixscale))

            starpos = locate_star(image, center, width, fwhm)

            try:
                im_crop = crop_image(image, starpos, im_size)

            except ValueError:
                warnings.warn("PSF size is too large to crop the image around the brightest "
                              "pixel (image index = "+str(self.m_count)+", pixel [x, y] = "
                              +str([starpos[0]]+[starpos[1]])+"). Using the center of the "
                              "image instead.")

                index.append(self.m_count)

                starpos = center_pixel(image)
                im_crop = crop_image(image, starpos, im_size)

            star.append((starpos[1], starpos[0]))

            self.m_count += 1

            return im_crop

        self.apply_function_to_images(_crop_around_star,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarExtractionModule...",
                                      func_args=(self.m_position,
                                                 self.m_image_size,
                                                 self.m_fwhm_star))

        history = "fwhm_star [pix] = "+str(self.m_fwhm_star)

        if self.m_index_out_port is not None:
            self.m_index_out_port.set_all(np.transpose(np.asarray(index)))
            self.m_index_out_port.copy_attributes(self.m_image_in_port)
            self.m_index_out_port.add_history("StarExtractionModule", history)

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("StarExtractionModule", history)
        self.m_image_out_port.add_attribute("STAR_POSITION", np.asarray(star), static=False)

        self.m_image_out_port.close_port()


class StarAlignmentModule(ProcessingModule):
    """
    Pipeline module to align the images with a cross-correlation in Fourier space.
    """

    def __init__(self,
                 name_in="star_align",
                 image_in_tag="im_arr",
                 ref_image_in_tag=None,
                 image_out_tag="im_arr_aligned",
                 interpolation="spline",
                 accuracy=10,
                 resize=None,
                 num_references=10,
                 subframe=None):
        """
        Constructor of StarAlignmentModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the stack of images that is read as input.
        ref_image_in_tag : str
            Tag of the database entry with the reference image(s) that are read as input. If it is
            set to None, a random subsample of *num_references* elements of *image_in_tag* is taken
            as reference images.
        image_out_tag : str
            Tag of the database entry with the images that are written as output.
        interpolation : str
            Type of interpolation that is used for shifting the images (spline, bilinear, or fft).
        accuracy : float
            Upsampling factor for the cross-correlation. Images will be registered to within
            1/accuracy of a pixel.
        resize : float
            Scaling factor for the up/down-sampling before the images are shifted.
        num_references : int
            Number of reference images for the cross-correlation.
        subframe : float
            Size (arcsec) of the subframe around the image center that is used for the
            cross-correlation. The full image is used if set to None.

        Returns
        -------
        NoneType
            None
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
        self.m_subframe = subframe

    def run(self):
        """
        Run method of the module. Applies a cross-correlation of the input images with respect to
        a stack of reference images, rescales the image dimensions, and shifts the images to a
        common center.

        Returns
        -------
        NoneType
            None
        """

        def _align_image(image_in):
            offset = np.array([0., 0.])

            for i in range(self.m_num_references):
                if self.m_subframe is None:
                    tmp_offset, _, _ = register_translation(ref_images[i, :, :],
                                                            image_in,
                                                            upsample_factor=self.m_accuracy)

                else:
                    sub_in = crop_image(image_in, None, self.m_subframe)
                    sub_ref = crop_image(ref_images[i, :, :], None, self.m_subframe)

                    tmp_offset, _, _ = register_translation(sub_ref,
                                                            sub_in,
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
                                    mode="reflect",
                                    anti_aliasing=True,
                                    multichannel=False)

                sum_after = np.sum(tmp_image)

                # Conserve flux because the rescale function normalizes all values to [0:1].
                tmp_image = tmp_image*(sum_before/sum_after)

            else:
                tmp_image = image_in

            return shift_image(tmp_image, offset, self.m_interpolation)

        if self.m_ref_image_in_port is not None:
            im_dim = self.m_ref_image_in_port.get_ndim()

            if im_dim == 3:
                if self.m_ref_image_in_port.get_shape()[0] > self.m_num_references:
                    ref_images = self.m_ref_image_in_port[np.sort(
                        np.random.choice(self.m_ref_image_in_port.get_shape()[0],
                                         self.m_num_references,
                                         replace=False)), :, :]

                else:
                    ref_images = self.m_ref_image_in_port.get_all()
                    self.m_num_references = self.m_ref_image_in_port.get_shape()[0]

            elif im_dim == 2:
                ref_images = np.array([self.m_ref_image_in_port.get_all(), ])
                self.m_num_references = 1

            else:
                raise ValueError("Reference images need to be 2D or 3D.")

        else:
            random = np.random.choice(self.m_image_in_port.get_shape()[0],
                                      self.m_num_references,
                                      replace=False)
            sort = np.sort(random)
            ref_images = self.m_image_in_port[sort, :, :]

        if self.m_subframe is not None:
            pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
            self.m_subframe = int(self.m_subframe/pixscale)

        self.apply_function_to_images(_align_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarAlignmentModule...")

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        if self.m_resize is not None:
            pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
            self.m_image_out_port.add_attribute("PIXSCALE", pixscale/self.m_resize)

        history = "resize = "+str(self.m_resize)
        self.m_image_out_port.add_history("StarAlignmentModule", history)
        self.m_image_out_port.close_port()


class StarCenteringModule(ProcessingModule):
    """
    Pipeline module for centering the star by fitting the PSF with a 2D Gaussian or Moffat
    function.
    """

    def __init__(self,
                 name_in="centering",
                 image_in_tag="im_arr",
                 image_out_tag="im_center",
                 mask_out_tag=None,
                 fit_out_tag="center_fit",
                 method="full",
                 interpolation="spline",
                 radius=0.1,
                 sign="positive",
                 model="gaussian",
                 filter_size=None,
                 **kwargs):
        """
        Constructor of StarCenteringModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with images that are read as input.
        image_out_tag : str
            Tag of the database entry with the centered images that are written as output. Should
            be different from *image_in_tag*. Data is not written when set to None.
        mask_out_tag : str
            Tag of the database entry with the masked images that are written as output. The
            unmasked part of the images is used for the fit. The effect of the smoothing that is
            applied by setting the *fwhm* parameter is also visible in the data of the
            *mask_out_tag*. Data is not written when set to None.
        fit_out_tag : str
            Tag of the database entry with the best-fit results of the model fit and the 1-sigma
            errors. Data is written in the following format: x offset (arcsec), x offset error
            (arcsec), y offset (arcsec), y offset error (arcsec), FWHM major axis (arcsec), FWHM
            major axis error (arcsec), FWHM minor axis (arcsec), FWHM minor axis error (arcsec),
            amplitude (counts), amplitude error (counts), angle (deg), angle error (deg) measured
            in counterclockwise direction with respect to the upward direction (i.e., East of
            North), offset (counts), offset error (counts), power index (only for Moffat function),
            and power index error (only for Moffat function).
        method : str
            Fit and shift all the images individually ("full") or only fit the mean of the cube and
            shift all images to that location ("mean"). The "mean" method could be used after
            running the :class:`~pynpoint.processing.centering.StarAlignmentModule`.
        interpolation : str
            Type of interpolation that is used for shifting the images (spline, bilinear, or fft).
        radius : float
            Radius (arcsec) around the center of the image beyond which pixels are neglected with
            the fit. The radius is centered on the position specified in *guess*, which is the
            center of the image by default.
        sign : str
            Fit a "positive" or "negative" Gaussian/Moffat. A negative model can be used to center
            coronagraphic data in which a dark hole is present.
        model : str
            Type of 2D model used to fit the PSF ("gaussian" or "moffat"). Both models are
            elliptical in shape.
        filter_size : float
            Standard deviation (arcsec) of the Gaussian filter that is used to smooth the
            images before fitting the model. No filter is applied if set to None.

        Keyword arguments
        -----------------
        guess : tuple(float, float, float, float, float, float, float, float)
            The initial parameter values for the least squares fit: x offset with respect to center
            (pix), y offset with respect to center (pix), FWHM x (pix), FWHM y (pix), amplitude
            (counts), angle (deg), offset (counts), and power index (only for Moffat function).

        Returns
        -------
        NoneType
            None
        """

        if "guess" in kwargs:
            self.m_guess = kwargs["guess"]

            if model == "gaussian" and len(self.m_guess) == 6:
                warnings.warn("The 'guess' parameter is missing one value. The guessed offset "
                              "value is set to zero.")

                self.m_guess = (self.m_guess[0], self.m_guess[1], self.m_guess[2], self.m_guess[3],
                                self.m_guess[4], self.m_guess[5], 0.)

        else:
            if model == "gaussian":
                self.m_guess = (0., 0., 1., 1., 1., 0., 0.)

            elif model == "moffat":
                self.m_guess = (0., 0., 1., 1., 1., 0., 0., 1.)

        super(StarCenteringModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if image_out_tag is None:
            self.m_image_out_port = None
        else:
            self.m_image_out_port = self.add_output_port(image_out_tag)

        if mask_out_tag is None:
            self.m_mask_out_port = None
        else:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)

        self.m_fit_out_port = self.add_output_port(fit_out_tag)

        self.m_method = method
        self.m_interpolation = interpolation
        self.m_radius = radius
        self.m_sign = sign
        self.m_model = model
        self.m_filter_size = filter_size
        self.m_model_func = None

        self.m_count = 0

    def run(self):
        """
        Run method of the module. Uses a non-linear least squares (Levenberg-Marquardt) to fit the
        the individual images or the mean of the stack with a 2D Gaussian or Moffat function,
        shifts the images with subpixel precision, and writes the centered images and the fitting
        results. The fitting results contain zeros in case the algorithm could not converge.

        Returns
        -------
        NoneType
            None
        """

        self.m_fit_out_port.del_all_data()
        self.m_fit_out_port.del_all_attributes()

        if self.m_image_out_port:
            self.m_image_out_port.del_all_data()
            self.m_image_out_port.del_all_attributes()

        if self.m_mask_out_port:
            self.m_mask_out_port.del_all_data()
            self.m_mask_out_port.del_all_attributes()

        npix = self.m_image_in_port.get_shape()[-1]
        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if self.m_radius:
            self.m_radius /= pixscale

        if self.m_filter_size:
            self.m_filter_size /= pixscale

        if npix%2 == 0:
            x_grid = y_grid = np.linspace(-npix/2+0.5, npix/2-0.5, npix)
            x_ap = np.linspace(-npix/2+0.5-self.m_guess[0], npix/2-0.5-self.m_guess[0], npix)
            y_ap = np.linspace(-npix/2+0.5-self.m_guess[1], npix/2-0.5-self.m_guess[1], npix)

        elif npix%2 == 1:
            x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)
            x_ap = np.linspace(-(npix-1)/2-self.m_guess[0], (npix-1)/2-self.m_guess[0], npix)
            y_ap = np.linspace(-(npix-1)/2-self.m_guess[1], (npix-1)/2-self.m_guess[1], npix)

        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        xx_ap, yy_ap = np.meshgrid(x_ap, y_ap)
        rr_ap = np.sqrt(xx_ap**2+yy_ap**2)

        def gaussian_2d(grid,
                        x_center,
                        y_center,
                        fwhm_x,
                        fwhm_y,
                        amp,
                        theta,
                        offset):
            """
            Function to create a 2D elliptical Gaussian model.

            Parameters
            ----------
            grid : numpy.ndarray
                Two 2D arrays with the mesh grid points in x and y direction.
            x_center : float
                Offset of the model center along the x axis (pix).
            y_center : float
                Offset of the model center along the y axis (pix).
            fwhm_x : float
                Full width at half maximum along the x axis (pix).
            fwhm_y : float
                Full width at half maximum along the y axis (pix).
            amp : float
                Peak flux.
            theta : float
                Rotation angle in counterclockwise direction (rad).
            offset : float
                Flux offset.

            Returns
            -------
            numpy.ndimage
                Raveled 2D elliptical Gaussian model.
            """

            (xx_grid, yy_grid) = grid

            x_diff = xx_grid - x_center
            y_diff = yy_grid - y_center

            sigma_x = fwhm_x/math.sqrt(8.*math.log(2.))
            sigma_y = fwhm_y/math.sqrt(8.*math.log(2.))

            a_gauss = 0.5 * ((np.cos(theta)/sigma_x)**2 + (np.sin(theta)/sigma_y)**2)
            b_gauss = 0.5 * ((np.sin(2.*theta)/sigma_x**2) - (np.sin(2.*theta)/sigma_y**2))
            c_gauss = 0.5 * ((np.sin(theta)/sigma_x)**2 + (np.cos(theta)/sigma_y)**2)

            gaussian = offset + amp*np.exp(-(a_gauss*x_diff**2 + b_gauss*x_diff*y_diff + \
                c_gauss*y_diff**2))

            if self.m_radius:
                gaussian = gaussian[rr_ap < self.m_radius]
            else:
                gaussian = np.ravel(gaussian)

            return gaussian

        def moffat_2d(grid,
                      x_center,
                      y_center,
                      fwhm_x,
                      fwhm_y,
                      amp,
                      theta,
                      offset,
                      beta):
            """
            Function to create a 2D elliptical Moffat model.

            Parameters
            ----------
            grid : numpy.ndarray
                Two 2D arrays with the mesh grid points in x and y direction.
            x_center : float
                Offset of the model center along the x axis (pix).
            y_center : float
                Offset of the model center along the y axis (pix).
            fwhm_x : float
                Full width at half maximum along the x axis (pix).
            fwhm_y : float
                Full width at half maximum along the y axis (pix).
            amp : float
                Peak flux.
            theta : float
                Rotation angle in counterclockwise direction (rad).
            offset : float
                Flux offset.
            beta : float
                Power index.

            Returns
            -------
            numpy.ndimage
                Raveled 2D elliptical Moffat model.
            """

            (xx_grid, yy_grid) = grid

            x_diff = xx_grid - x_center
            y_diff = yy_grid - y_center

            alpha_x = 0.5*fwhm_x/np.sqrt(2.**(1./beta)-1.)
            alpha_y = 0.5*fwhm_y/np.sqrt(2.**(1./beta)-1.)

            a_moffat = (np.cos(theta)/alpha_x)**2. + (np.sin(theta)/alpha_y)**2.
            b_moffat = (np.sin(theta)/alpha_x)**2. + (np.cos(theta)/alpha_y)**2.
            c_moffat = 2.*np.sin(theta)*np.cos(theta)*(1./alpha_x**2. - 1./alpha_y**2.)

            a_term = a_moffat*x_diff**2
            b_term = b_moffat*y_diff**2
            c_term = c_moffat*x_diff*y_diff

            moffat = offset + amp / (1.+a_term+b_term+c_term)**beta

            if self.m_radius:
                moffat = moffat[rr_ap < self.m_radius]
            else:
                moffat = np.ravel(moffat)

            return moffat

        def _least_squares(image):
            if self.m_filter_size:
                image = gaussian_filter(image, self.m_filter_size)

            if self.m_mask_out_port:
                mask = np.copy(image)

                if self.m_radius:
                    mask[rr_ap > self.m_radius] = 0.

                if self.m_method == "mean":
                    self.m_mask_out_port.set_all(mask)
                elif self.m_method == "full":
                    self.m_mask_out_port.append(mask, data_dim=3)

            if self.m_sign == "negative":
                image = -image + np.abs(np.min(-image))

            if self.m_radius:
                image = image[rr_ap < self.m_radius]
            else:
                image = np.ravel(image)

            if self.m_model == "gaussian":
                self.m_model_func = gaussian_2d

            elif self.m_model == "moffat":
                self.m_model_func = moffat_2d

            try:
                popt, pcov = curve_fit(self.m_model_func,
                                       (xx_grid, yy_grid),
                                       image,
                                       p0=self.m_guess,
                                       sigma=None,
                                       method='lm')

                perr = np.sqrt(np.diag(pcov))

            except RuntimeError:
                if self.m_model == "gaussian":
                    popt = np.zeros(7)
                    perr = np.zeros(7)

                elif self.m_model == "moffat":
                    popt = np.zeros(8)
                    perr = np.zeros(8)

                self.m_count += 1

            if self.m_model == "gaussian":

                res = np.asarray((popt[0]*pixscale, perr[0]*pixscale,
                                  popt[1]*pixscale, perr[1]*pixscale,
                                  popt[2]*pixscale, perr[2]*pixscale,
                                  popt[3]*pixscale, perr[3]*pixscale,
                                  popt[4], perr[4],
                                  math.degrees(popt[5])%360., math.degrees(perr[5]),
                                  popt[6], perr[6]))

            elif self.m_model == "moffat":

                res = np.asarray((popt[0]*pixscale, perr[0]*pixscale,
                                  popt[1]*pixscale, perr[1]*pixscale,
                                  popt[2]*pixscale, perr[2]*pixscale,
                                  popt[3]*pixscale, perr[3]*pixscale,
                                  popt[4], perr[4],
                                  math.degrees(popt[5])%360., math.degrees(perr[5]),
                                  popt[6], perr[6],
                                  popt[7], perr[7]))

            self.m_fit_out_port.append(res, data_dim=2)

            return popt

        def _centering(image,
                       popt):

            if self.m_method == "full":
                popt = _least_squares(np.copy(image))

            return shift_image(image, (-popt[1], -popt[0]), self.m_interpolation)

        ndim = self.m_image_in_port.get_ndim()
        npix = self.m_image_in_port.get_shape()[-1]

        nimages = self.m_image_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        if self.m_method == "full":
            popt = None

        elif self.m_method == "mean":
            im_mean = np.zeros((npix, npix))
            for i, _ in enumerate(frames[:-1]):
                im_mean += np.sum(self.m_image_in_port[frames[i]:frames[i+1], ], axis=0)

            popt = _least_squares(im_mean/float(nimages))

        self.apply_function_to_images(_centering,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarCenteringModule...",
                                      func_args=(popt, ))

        if self.m_count > 0:
            print("Fit could not converge on %s image(s). [WARNING]" % self.m_count)

        history = "method = "+self.m_method

        if self.m_image_out_port:
            self.m_image_out_port.copy_attributes(self.m_image_in_port)
            self.m_image_out_port.add_history("StarCenteringModule", history)

        self.m_fit_out_port.copy_attributes(self.m_image_in_port)
        self.m_fit_out_port.add_history("StarCenteringModule", history)

        if self.m_mask_out_port:
            self.m_mask_out_port.copy_attributes(self.m_image_in_port)
            self.m_mask_out_port.add_history("StarCenteringModule", history)

        self.m_fit_out_port.close_port()


class ShiftImagesModule(ProcessingModule):
    """
    Pipeline module for shifting a stack of images.
    """

    def __init__(self,
                 shift_xy,
                 interpolation="spline",
                 name_in="shift",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_shifted"):
        """
        Constructor of ShiftImagesModule.

        Parameters
        ----------
        shift_xy : tuple(float, float)
            The shift (pix) in x and y direction as (delta_x, delta_y).
        interpolation : str
            Type of interpolation that is used for shifting the images (spline, bilinear, or fft).
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(ShiftImagesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_shift = (shift_xy[1], shift_xy[0])
        self.m_interpolation = interpolation

    def run(self):
        """
        Run method of the module. Shifts an image with a fifth order spline, bilinear, or a
        Fourier shift interpolation.

        Returns
        -------
        NoneType
            None
        """

        def _image_shift(image, shift, interpolation):

            return shift_image(image, shift, interpolation)

        self.apply_function_to_images(_image_shift,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running ShiftImagesModule...",
                                      func_args=(self.m_shift, self.m_interpolation))

        history = "shift_xy = "+str(self.m_shift)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("ShiftImagesModule", history)
        self.m_image_out_port.close_port()


class WaffleCenteringModule(ProcessingModule):
    """
    Pipeline module for centering of SPHERE data obtained with a Lyot coronagraph for which center
    frames with satellite spots are available.
    """

    __author__ = "Alexander Bohn"

    def __init__(self,
                 name_in="center_images",
                 image_in_tag="im_arr",
                 center_in_tag="center_frame",
                 image_out_tag="im_arr_centered_cut",
                 size=2.,
                 center=None,
                 radius=45.,
                 pattern="x",
                 sigma=0.06,
                 dither=False):
        """
        Constructor of WaffleCenteringModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with science images that are read as input.
        center_in_tag : str
            Tag of the database entry with the center frame that is read as input.
        image_out_tag : str
            Tag of the database entry with the centered images that are written as output. Should
            be different from *image_in_tag*.
        size : float
            Image size (arcsec) for both dimensions. Original image size is used if set to None.
        center : tuple(float, float)
            Approximate position (x0, y0) of the coronagraph. The center of the image is used if
            set to None.
        radius : float
            Approximate separation (pix) of the waffle spots from the star.
        pattern : str
            Waffle pattern that is used ("x" or "+").
        sigma : float
            Standard deviation (arcsec) of the Gaussian kernel that is used for the unsharp
            masking.
        dither : bool
            Apply dithering correction based on the DITHER_X and DITHER_Y attributes.

        Returns
        -------
        NoneType
            None
        """

        super(WaffleCenteringModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_center_in_port = self.add_input_port(center_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_size = size
        self.m_center = center
        self.m_radius = radius
        self.m_pattern = pattern
        self.m_sigma = sigma
        self.m_dither = dither

    def run(self):
        """
        Run method of the module. Locates the position of the calibration spots in the center
        frame. From the four spots, the position of the star behind the coronagraph is fitted,
        and the images are shifted and cropped.

        Returns
        -------
        NoneType
            None
        """

        def _get_center(ndim, center):
            center_frame = self.m_center_in_port[0, ]

            if center_shape[0] > 1:
                warnings.warn("Multiple center images found. Using the first image of the stack.")

            if center is None:
                center = center_pixel(center_frame)
            else:
                center = (np.floor(center[0]), np.floor(center[1]))

            return center_frame, center

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        center_ndim = self.m_center_in_port.get_ndim()
        center_shape = self.m_center_in_port.get_shape()
        im_shape = self.m_image_in_port.get_shape()

        center_frame, self.m_center = _get_center(center_ndim, self.m_center)

        if im_shape[-2:] != center_shape[-2:]:
            raise ValueError("Science and center images should have the same shape.")

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        self.m_sigma /= pixscale
        if self.m_size is not None:
            self.m_size = int(math.ceil(self.m_size/pixscale))

        if self.m_dither:
            dither_x = self.m_image_in_port.get_attribute("DITHER_X")
            dither_y = self.m_image_in_port.get_attribute("DITHER_Y")

            nframes = self.m_image_in_port.get_attribute("NFRAMES")
            nframes = np.cumsum(nframes)
            nframes = np.insert(nframes, 0, 0)

        center_frame_unsharp = center_frame - gaussian_filter(input=center_frame,
                                                              sigma=self.m_sigma)

        # size of center image, only works with odd value
        ref_image_size = 21

        # Arrays for the positions
        x_pos = np.zeros(4)
        y_pos = np.zeros(4)

        # Loop for 4 waffle spots
        for i in range(4):
            # Approximate positions of waffle spots
            if self.m_pattern == "x":
                x_0 = np.floor(self.m_center[0] + self.m_radius * np.cos(np.pi / 4. * (2 * i + 1)))
                y_0 = np.floor(self.m_center[1] + self.m_radius * np.sin(np.pi / 4. * (2 * i + 1)))

            elif self.m_pattern == "+":
                x_0 = np.floor(self.m_center[0] + self.m_radius * np.cos(np.pi / 4. * (2 * i)))
                y_0 = np.floor(self.m_center[1] + self.m_radius * np.sin(np.pi / 4. * (2 * i)))

            tmp_center_frame = crop_image(image=center_frame_unsharp,
                                          center=(int(y_0), int(x_0)),
                                          size=ref_image_size)

            # find maximum in tmp image
            y_max, x_max = np.unravel_index(np.argmax(tmp_center_frame),
                                            dims=tmp_center_frame.shape)

            pixmax = tmp_center_frame[y_max, x_max]
            max_pos = np.array([x_max, y_max]).reshape(1, 2)

            # Check whether it is the correct maximum: second brightest pixel should be nearby
            tmp_center_frame[y_max, x_max] = 0.

            # introduce distance parameter
            dist = np.inf

            while dist > 2:
                y_max_new, x_max_new = np.unravel_index(np.argmax(tmp_center_frame),
                                                        dims=tmp_center_frame.shape)

                pixmax_new = tmp_center_frame[y_max_new, x_max_new]

                # Caculate minimal distance to previous points
                tmp_center_frame[y_max_new, x_max_new] = 0.

                dist = np.amin(np.linalg.norm(np.vstack((max_pos[:, 0]-x_max_new,
                                                         max_pos[:, 1]-y_max_new)),
                                              axis=0))

                if dist <= 2 and pixmax_new < pixmax:
                    break

                max_pos = np.vstack((max_pos, [x_max_new, y_max_new]))

                x_max = x_max_new
                y_max = y_max_new
                pixmax = pixmax_new

            x_0 = x_0 - (ref_image_size-1)/2 + x_max
            y_0 = y_0 - (ref_image_size-1)/2 + y_max

            # create reference image around determined maximum
            ref_center_frame = crop_image(image=center_frame_unsharp,
                                          center=(int(y_0), int(x_0)),
                                          size=ref_image_size)

            # Fit the data using astropy.modeling
            gauss_init = models.Gaussian2D(amplitude=np.amax(ref_center_frame),
                                           x_mean=x_0,
                                           y_mean=y_0,
                                           x_stddev=1.,
                                           y_stddev=1.,
                                           theta=0.)

            fit_gauss = fitting.LevMarLSQFitter()

            y_grid, x_grid = np.mgrid[y_0-(ref_image_size-1)/2:y_0+(ref_image_size-1)/2+1,
                                      x_0-(ref_image_size-1)/2:x_0+(ref_image_size-1)/2+1]

            gauss = fit_gauss(gauss_init,
                              x_grid,
                              y_grid,
                              ref_center_frame)

            x_pos[i] = gauss.x_mean.value
            y_pos[i] = gauss.y_mean.value

        # Find star position as intersection of two lines

        x_center = ((y_pos[0]-x_pos[0]*(y_pos[2]-y_pos[0])/(x_pos[2]-float(x_pos[0]))) - \
                    (y_pos[1]-x_pos[1]*(y_pos[1]-y_pos[3])/(x_pos[1]-float(x_pos[3])))) / \
                   ((y_pos[1]-y_pos[3])/(x_pos[1]-float(x_pos[3])) - \
                    (y_pos[2]-y_pos[0])/(x_pos[2]-float(x_pos[0])))

        y_center = x_center*(y_pos[1]-y_pos[3])/(x_pos[1]-float(x_pos[3])) + \
                   (y_pos[1]-x_pos[1]*(y_pos[1]-y_pos[3])/(x_pos[1]-float(x_pos[3])))

        nimages = self.m_image_in_port.get_shape()[0]
        npix = self.m_image_in_port.get_shape()[1]

        for i in range(nimages):
            progress(i, nimages, "Running WaffleCenteringModule...")

            image = self.m_image_in_port[i, ]

            shift_yx = [(float(im_shape[-2])-1.)/2. - y_center,
                        (float(im_shape[-1])-1.)/2. - x_center]

            if self.m_dither:
                index = np.digitize(i, nframes, right=False) - 1

                shift_yx[0] -= dither_y[index]
                shift_yx[1] -= dither_x[index]

            if npix%2 == 0 and self.m_size is not None:
                im_tmp = np.zeros((image.shape[0]+1, image.shape[1]+1))
                im_tmp[:-1, :-1] = image
                image = im_tmp

                shift_yx[0] += 0.5
                shift_yx[1] += 0.5

            im_shift = shift_image(image, shift_yx, "spline")

            if self.m_size is not None:
                im_crop = crop_image(im_shift, None, self.m_size)
                self.m_image_out_port.append(im_crop, data_dim=3)
            else:
                self.m_image_out_port.append(im_shift, data_dim=3)

        sys.stdout.write("Running WaffleCenteringModule... [DONE]\n")
        sys.stdout.write("Center [x, y] = ["+str(x_center)+", "+str(y_center)+"]\n")
        sys.stdout.flush()

        history = "position [x, y] = "+str([round(x_center, 4), round(y_center, 4)])
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("WaffleCenteringModule", history)
        self.m_image_out_port.close_port()
