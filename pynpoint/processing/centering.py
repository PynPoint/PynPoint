"""
Pipeline modules for aligning and centering of the star.
"""

import math
import time
import warnings

from typing import Optional, Tuple, Union

import numpy as np

from astropy.modeling import fitting, models
from scipy.ndimage.filters import gaussian_filter
from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import center_pixel, crop_image, pixel_distance, shift_image, \
                                subpixel_distance
from pynpoint.util.module import memory_frames, progress
from pynpoint.util.apply_func import align_image, apply_shift, fit_2d_function


class StarAlignmentModule(ProcessingModule):
    """
    Pipeline module to align the images with a cross-correlation in Fourier space.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 ref_image_in_tag: Optional[str] = None,
                 interpolation: str = 'spline',
                 accuracy: float = 10.,
                 resize: Optional[float] = None,
                 num_references: int = 10,
                 subframe: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the stack of images that is read as input.
        ref_image_in_tag : str, None
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
        resize : float, None
            Scaling factor for the up/down-sampling before the images are shifted. Not used if set
            to None.
        num_references : int
            Number of reference images for the cross-correlation.
        subframe : float, None
            Size (arcsec) of the subframe around the image center that is used for the
            cross-correlation. The full image is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

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

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Applies a cross-correlation of the input images with respect to
        a stack of reference images, rescales the image dimensions, and shifts the images to a
        common center.

        Returns
        -------
        NoneType
            None
        """

        if self.m_ref_image_in_port is None:
            random = np.random.choice(self.m_image_in_port.get_shape()[0],
                                      self.m_num_references,
                                      replace=False)

            ref_images = self.m_image_in_port[np.sort(random), :, :]

        else:
            n_ref = self.m_ref_image_in_port.get_shape()[0]

            if n_ref < self.m_num_references:
                warnings.warn(f'Number of available images ({n_ref}) is smaller than '
                              f'num_references ({self.m_num_references}). Using all '
                              f'available images instead.')

                self.m_num_references = n_ref

            ref_index = np.sort(np.random.choice(n_ref, self.m_num_references, replace=False))
            ref_images = self.m_ref_image_in_port[ref_index, :, :]

        if self.m_subframe is not None:
            pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
            self.m_subframe = int(self.m_subframe/pixscale)

        self.apply_function_to_images(align_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Aligning images',
                                      func_args=(self.m_interpolation,
                                                 self.m_accuracy,
                                                 self.m_resize,
                                                 self.m_num_references,
                                                 self.m_subframe,
                                                 ref_images.reshape(-1),
                                                 ref_images.shape))

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        if self.m_resize is not None:
            pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
            new_pixscale = pixscale/self.m_resize
            self.m_image_out_port.add_attribute('PIXSCALE', new_pixscale)
            print(f'New pixel scale (arcsec) = {new_pixscale:.2f}')

        history = f'resize = {self.m_resize}'
        self.m_image_out_port.add_history('StarAlignmentModule', history)
        self.m_image_out_port.close_port()


class FitCenterModule(ProcessingModule):
    """
    Pipeline module for fitting the PSF with a 2D Gaussian or Moffat function.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 fit_out_tag: str,
                 mask_out_tag: Optional[str] = None,
                 method: str = 'full',
                 mask_radii: Tuple[Optional[float], float] = (None, 0.1),
                 sign: str = 'positive',
                 model: str = 'gaussian',
                 filter_size: Optional[float] = None,
                 **kwargs: Union[Tuple[float, float, float, float, float, float, float],
                                 Tuple[float, float, float, float, float, float, float, float],
                                 float]) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Database tag of the images that are read as input.
        fit_out_tag : str
            Database tag where the best-fit results and 1σ errors will be stored.
            The data are written in the following format: x offset (pix), x offset error (pix)
            y offset (pix), y offset error (pix), FWHM major axis (arcsec), FWHM major axis error
            (arcsec), FWHM minor axis (arcsec), FWHM minor axis error (arcsec), amplitude (ADU),
            amplitude error (ADU), angle (deg), angle error (deg) measured in counterclockwise
            direction with respect to the upward direction (i.e. east of north), offset (ADU),
            offset error (ADU), power index (only for Moffat function), and power index error
            (only for Moffat function). The ``fit_out_tag`` can be used as argument of ``shift_xy``
            when running the :class:`~pynpoint.processing.centering.ShiftImagesModule`.
        mask_out_tag : str, None
            Database tag where the masked images will be stored. The unmasked part of the images is
            used for the fit. The effect of the smoothing that is applied by setting the ``fwhm``
            argument is also visible in the data of the ``mask_out_tag``. The data are not stored
            if the argument is set to None. The :class:`~pynpoint.core.dataio.OutputPort` of
            ``mask_out_tag`` can only be used when ``CPU = 1``.
        method : str
            Fit and shift each image individually ('full') or only fit the mean of the cube and
            shift each image by this constant offset ('mean'). The 'mean' method can be used in
            case the images are already aligned with
            :class:`~pynpoint.processing.centering.StarAlignmentModule`.
        mask_radii : tuple(float, float), tuple(None, float)
            Inner and outer radius (arcsec) within and beyond which pixels are neglected during the
            fit. The radii are centered at the position that specified with the argument of
            ``guess``, which is the center of the image by default. The outer mask (second value
            of ``mask_radii``) is mandatory whereas radius of the inner mask is optional and can
            be set to None.
        sign : str
            Fit a 'positive' or 'negative' Gaussian/Moffat function. A 'negative' model can be used
            to center coronagraphic data in which a dark hole.
        model : str
            Type of 2D model that is used for the fit ('gaussian' or 'moffat'). Both models are
            elliptical in shape.
        filter_size : float, None
            Standard deviation (arcsec) of the Gaussian filter that is used to smooth the
            images before fitting the model. No filter is applied if set to None.

        Keyword arguments
        -----------------
        guess : tuple(float, float, float, float, float, float, float, float),
                tuple(float, float, float, float, float, float, float, float, float)
            The initial parameter values for the least squares fit: x offset with respect to center
            (pix), y offset with respect to center (pix), FWHM x (pix), FWHM y (pix), amplitude
            (ADU), angle (deg), offset (ADU), and power index (only for Moffat function).

        Returns
        -------
        NoneType
            None
        """

        if 'guess' in kwargs:
            self.m_guess = kwargs['guess']

        else:
            if model == 'gaussian':
                self.m_guess = (0., 0., 1., 1., 1., 0., 0.)

            elif model == 'moffat':
                self.m_guess = (0., 0., 1., 1., 1., 0., 0., 1.)

        if 'radius' in kwargs:
            mask_radii = (None, kwargs['radius'])

            warnings.warn(f'The \'radius\' parameter has been deprecated. Please use the '
                          f'\'mask_radii\' parameter instead. The argument of \'mask_radii\' '
                          f'is set to {mask_radii}.', DeprecationWarning)

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_fit_out_port = self.add_output_port(fit_out_tag)

        if mask_out_tag is None:
            self.m_mask_out_port = None
        else:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)

        self.m_method = method
        self.m_mask_radii = mask_radii
        self.m_sign = sign
        self.m_model = model
        self.m_filter_size = filter_size

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Uses a non-linear least squares (Levenberg-Marquardt) method
        to fit the the individual images or the mean of all images with a 2D Gaussian or Moffat
        function. The best-fit results and errors are stored and contain zeros in case the
        algorithm could not converge. The ``fit_out_tag`` can be used as argument of ``shift_xy``
        when running the :class:`~pynpoint.processing.centering.ShiftImagesModule`.

        Returns
        -------
        NoneType
            None
        """

        memory = self._m_config_port.get_attribute('MEMORY')
        cpu = self._m_config_port.get_attribute('CPU')
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        if cpu > 1 and self.m_mask_out_port is not None:
            warnings.warn('The mask_out_port can only be used if CPU=1. No data will be '
                          'stored to this output port.')

            del self._m_output_ports[self.m_mask_out_port.tag]
            self.m_mask_out_port = None

        if self.m_mask_radii[0] is None:
            # Convert from arcsec to pixels and change None to 0
            self.m_mask_radii = (0., self.m_mask_radii[1]/pixscale)

        else:
            # Convert from arcsec to pixels
            self.m_mask_radii = (self.m_mask_radii[0]/pixscale, self.m_mask_radii[1]/pixscale)

        if self.m_filter_size:
            # Convert from arcsec to pixels
            self.m_filter_size /= pixscale

        _, xx_grid, yy_grid = pixel_distance(self.m_image_in_port.get_shape()[-2:], position=None)

        rr_ap = subpixel_distance(self.m_image_in_port.get_shape()[-2:],
                                  position=(self.m_guess[1], self.m_guess[0]),
                                  shift_center=False)  # (y, x)

        nimages = self.m_image_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        if self.m_method == 'full':

            self.apply_function_to_images(fit_2d_function,
                                          self.m_image_in_port,
                                          self.m_fit_out_port,
                                          'Fitting the stellar PSF',
                                          func_args=(self.m_mask_radii,
                                                     self.m_sign,
                                                     self.m_model,
                                                     self.m_filter_size,
                                                     self.m_guess,
                                                     self.m_mask_out_port,
                                                     xx_grid,
                                                     yy_grid,
                                                     rr_ap,
                                                     pixscale))

        elif self.m_method == 'mean':
            print('Fitting the stellar PSF...', end='')

            im_mean = np.zeros(self.m_image_in_port.get_shape()[1:3])

            for i, _ in enumerate(frames[:-1]):
                im_mean += np.sum(self.m_image_in_port[frames[i]:frames[i+1], ], axis=0)

            best_fit = fit_2d_function(im_mean/float(nimages),
                                       0,
                                       self.m_mask_radii,
                                       self.m_sign,
                                       self.m_model,
                                       self.m_filter_size,
                                       self.m_guess,
                                       self.m_mask_out_port,
                                       xx_grid,
                                       yy_grid,
                                       rr_ap,
                                       pixscale)

            best_fit = best_fit[np.newaxis, ...]
            best_fit = np.repeat(best_fit, nimages, axis=0)

            self.m_fit_out_port.set_all(best_fit, data_dim=2)

            print(' [DONE]')

        history = f'model = {self.m_model}'

        self.m_fit_out_port.copy_attributes(self.m_image_in_port)
        self.m_fit_out_port.add_history('FitCenterModule', history)

        if self.m_mask_out_port:
            self.m_mask_out_port.copy_attributes(self.m_image_in_port)
            self.m_mask_out_port.add_history('FitCenterModule', history)

        self.m_fit_out_port.close_port()


class ShiftImagesModule(ProcessingModule):
    """
    Pipeline module for shifting a stack of images.
    """

    __author__ = 'Tomas Stolker, Benedikt Schmidhuber'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 shift_xy: Union[Tuple[float, float], str],
                 interpolation: str = 'spline') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        shift_xy : tuple(float, float), str
            The shift (pix) in x and y direction as (delta_x, delta_y). Or, a database tag with
            the fit results from the :class:`~pynpoint.processing.centering.FitCenterModule`.
        interpolation : str
            Interpolation type for shifting of the images ('spline', 'bilinear', or 'fft').

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_interpolation = interpolation

        if isinstance(shift_xy, str):
            self.m_fit_in_port = self.add_input_port(shift_xy)
            self.m_shift = None

        else:
            self.m_fit_in_port = None
            self.m_shift = (shift_xy[1], shift_xy[0])

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Shifts an image with a fifth order spline, bilinear, or a
        Fourier shift interpolation.

        Returns
        -------
        NoneType
            None
        """

        constant = True

        # read the fit results from the self.m_fit_in_port if available
        if self.m_fit_in_port is not None:

            self.m_shift = -1.*self.m_fit_in_port[:, [0, 2]]  # (x, y)
            self.m_shift = self.m_shift[:, [1, 0]]  # (y, x)

            # check if data in self.m_fit_in_port is constant for all images using the
            # constant flag
            if not np.allclose(self.m_fit_in_port.get_all() - self.m_fit_in_port[0, ], 0.0):
                constant = False

            if constant:
                # if the offset is constant then use the first element for all images
                self.m_shift = self.m_shift[0, ]

            else:
                # if the offset is not constant, then apply the shifts to each frame individually
                for i, shift in enumerate(self.m_shift):
                    shifted_image = shift_image(self.m_image_in_port[i, ],
                                                shift,
                                                self.m_interpolation)

                    # append the shifted images to the selt.m_image_out_port database entry
                    self.m_image_out_port.append(shifted_image, data_dim=3)

                mean_shift = np.mean(self.m_shift, axis=0)
                history = f'shift_xy = {mean_shift[0]:.2f}, {mean_shift[1]:.2f}'

        # apply a constant shift
        if constant:

            self.apply_function_to_images(apply_shift,
                                          self.m_image_in_port,
                                          self.m_image_out_port,
                                          'Shifting the images',
                                          func_args=(self.m_shift,
                                                     self.m_interpolation))

            # if self.m_fit_in_port is None or constant:
            history = f'shift_xy = {self.m_shift[0]:.2f}, {self.m_shift[1]:.2f}'

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('ShiftImagesModule', history)
        self.m_image_out_port.close_port()


class WaffleCenteringModule(ProcessingModule):
    """
    Pipeline module for centering of coronagraphic data for which dedicated center frames with
    satellite spots are available.
    """

    __author__ = 'Alexander Bohn'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 center_in_tag: str,
                 image_out_tag: str,
                 size: Optional[float] = None,
                 center: Optional[Tuple[float, float]] = None,
                 radius: float = 45.,
                 pattern: str = None,
                 angle: float = 45.,
                 sigma: float = 0.06,
                 dither: bool = False) -> None:
        """
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
        size : float, None
            Image size (arcsec) for both dimensions. Original image size is used if set to None.
        center : tuple(float, float), None
            Approximate position (x0, y0) of the coronagraph. The center of the image is used if
            set to None.
        radius : float
            Approximate separation (pix) of the satellite spots from the star. For IFS data, the
            separation of the spots in the image with the shortest wavelength is required.
        pattern : str, None
            Waffle pattern that is used ('x' or '+'). This parameter will be deprecated in a future
            release. Please use the ``angle`` parameter instead. The parameter will be ignored if
            set to None.
        angle : float
            Angle offset (deg) in clockwise direction of the satellite spots with respect to the
            '+' orientation (i.e. when the spots are located along the horizontal and vertical
            axis). The previous use of the '+' pattern corresponds to 0 degrees and 'x' pattern
            corresponds to 45 degrees. SPHERE/IFS data requires an angle of 55.48 degrees.
        sigma : float
            Standard deviation (arcsec) of the Gaussian kernel that is used for the unsharp
            masking.
        dither : bool
            Apply dithering correction based on the ``DITHER_X`` and ``DITHER_Y`` attributes.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_center_in_port = self.add_input_port(center_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_size = size
        self.m_center = center
        self.m_radius = radius
        self.m_pattern = pattern
        self.m_angle = angle
        self.m_sigma = sigma
        self.m_dither = dither

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Locates the position of the calibration spots in the center
        frame. From the four spots, the position of the star behind the coronagraph is fitted,
        and the images are shifted and cropped.

        Returns
        -------
        NoneType
            None
        """

        @typechecked
        def _get_center(image_number: int,
                        center: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Tuple[int, int]]:

            if center_shape[-3] > 1:
                warnings.warn('Multiple center images found. Using the first image of the stack.')

            if ndim == 3:
                center_frame = self.m_center_in_port[0, ]
            elif ndim == 4:
                center_frame = self.m_center_in_port[image_number, 0, ]

            if center is None:
                center = center_pixel(center_frame)
            else:
                center = (int(np.floor(center[0])), int(np.floor(center[1])))

            return center_frame, center

        center_shape = self.m_center_in_port.get_shape()
        im_shape = self.m_image_in_port.get_shape()
        ndim = self.m_image_in_port.get_ndim()

        center_frame, self.m_center = _get_center(0, self.m_center)

        # Read in wavelength information or set it to default values
        if ndim == 4:
            wavelength = self.m_image_in_port.get_attribute('WAVELENGTH')

            if wavelength is None:
                raise ValueError('The wavelength information is required to centre IFS data. '
                                 'Please add it via the WavelengthReadingModule before using '
                                 'the WaffleCenteringModule.')

            if im_shape[0] != center_shape[0]:
                raise ValueError(f'Number of science wavelength channels: {im_shape[0]}. '
                                 f'Number of center wavelength channels: {center_shape[0]}. '
                                 'Exactly one center image per wavelength is required.')

            wavelength_min = np.min(wavelength)

        elif ndim == 3:
            # for none ifs data, use default value
            wavelength = [1.]
            wavelength_min = 1.

        # check if science and center images have the same shape
        if im_shape[-2:] != center_shape[-2:]:
            raise ValueError('Science and center images should have the same shape.')

        # Setting angle via pattern (used for backwards compability)
        if self.m_pattern is not None:

            if self.m_pattern == 'x':
                self.m_angle = 45.

            elif self.m_pattern == '+':
                self.m_angle = 0.

            else:
                raise ValueError(f'The pattern {self.m_pattern} is not valid. Please select '
                                 f'either \'x\' or \'+\'.')

            warnings.warn(f'The \'pattern\' parameter will be deprecated in a future release. '
                          f'Please Use the \'angle\' parameter instead and set it to '
                          f'{self.m_angle} degrees.',
                          DeprecationWarning)

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        self.m_sigma /= pixscale

        if self.m_size is not None:
            self.m_size = int(math.ceil(self.m_size/pixscale))

        if self.m_dither:
            dither_x = self.m_image_in_port.get_attribute('DITHER_X')
            dither_y = self.m_image_in_port.get_attribute('DITHER_Y')

            nframes = self.m_image_in_port.get_attribute('NFRAMES')
            nframes = np.cumsum(nframes)
            nframes = np.insert(nframes, 0, 0)

        # size of center image, only works with odd value
        ref_image_size = 21

        # Arrays for the positions
        x_pos = np.zeros(4)
        y_pos = np.zeros(4)

        # Arrays for the center position for each wavelength
        x_center = np.zeros((len(wavelength)))
        y_center = np.zeros((len(wavelength)))

        # Loop for 4 waffle spots
        for w, wave_nr in enumerate(wavelength):

            # Prapre centering frame
            center_frame, _ = _get_center(w, self.m_center)

            center_frame_unsharp = center_frame - gaussian_filter(input=center_frame,
                                                                  sigma=self.m_sigma)

            for i in range(4):
                # Approximate positions of waffle spots
                radius = self.m_radius * wave_nr / wavelength_min

                x_0 = np.floor(self.m_center[0] + radius *
                               np.cos(self.m_angle*np.pi/180 + np.pi / 4. * (2 * i)))

                y_0 = np.floor(self.m_center[1] + radius *
                               np.sin(self.m_angle*np.pi/180 + np.pi / 4. * (2 * i)))

                tmp_center_frame = crop_image(image=center_frame_unsharp,
                                              center=(int(y_0), int(x_0)),
                                              size=ref_image_size)

                # find maximum in tmp image
                coords = np.unravel_index(indices=np.argmax(tmp_center_frame),
                                          shape=tmp_center_frame.shape)

                y_max, x_max = coords[0], coords[1]

                pixmax = tmp_center_frame[y_max, x_max]
                max_pos = np.array([x_max, y_max]).reshape(1, 2)

                # Check whether it is the correct maximum: second brightest pixel should be nearby
                tmp_center_frame[y_max, x_max] = 0.

                # introduce distance parameter
                dist = np.inf

                while dist > 2:
                    coords = np.unravel_index(indices=np.argmax(tmp_center_frame),
                                              shape=tmp_center_frame.shape)

                    y_max_new, x_max_new = coords[0], coords[1]

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

            x_center[w] = ((y_pos[0]-x_pos[0]*(y_pos[2]-y_pos[0])/(x_pos[2]-float(x_pos[0]))) -
                           (y_pos[1]-x_pos[1]*(y_pos[1]-y_pos[3])/(x_pos[1]-float(x_pos[3])))) / \
                          ((y_pos[1]-y_pos[3])/(x_pos[1]-float(x_pos[3])) -
                           (y_pos[2]-y_pos[0])/(x_pos[2]-float(x_pos[0])))

            y_center[w] = x_center[w]*(y_pos[1]-y_pos[3])/(x_pos[1]-float(x_pos[3])) + \
                (y_pos[1]-x_pos[1]*(y_pos[1]-y_pos[3])/(x_pos[1]-float(x_pos[3])))

        # Adjust science images
        nimages = self.m_image_in_port.get_shape()[-3]
        npix = self.m_image_in_port.get_shape()[-2]
        nwavelengths = len(wavelength)

        start_time = time.time()

        for i in range(nimages):
            im_storage = []
            for j in range(nwavelengths):
                im_index = i*nwavelengths + j

                progress(im_index, nimages*nwavelengths, 'Centering the images...', start_time)

                if ndim == 3:
                    image = self.m_image_in_port[i, ]
                elif ndim == 4:
                    image = self.m_image_in_port[j, i, ]

                shift_yx = np.array([(float(im_shape[-2])-1.)/2. - y_center[j],
                                     (float(im_shape[-1])-1.)/2. - x_center[j]])

                if self.m_dither:
                    index = np.digitize(i, nframes, right=False) - 1

                    shift_yx[0] -= dither_y[index]
                    shift_yx[1] -= dither_x[index]

                if npix % 2 == 0 and self.m_size is not None:
                    im_tmp = np.zeros((image.shape[0]+1, image.shape[1]+1))
                    im_tmp[:-1, :-1] = image
                    image = im_tmp

                    shift_yx[0] += 0.5
                    shift_yx[1] += 0.5

                im_shift = shift_image(image, shift_yx, 'spline')

                if self.m_size is not None:
                    im_crop = crop_image(im_shift, None, self.m_size)
                    im_storage.append(im_crop)
                else:
                    im_storage.append(im_shift)

            if ndim == 3:
                self.m_image_out_port.append(im_storage[0], data_dim=3)
            elif ndim == 4:
                self.m_image_out_port.append(np.asarray(im_storage), data_dim=4)

        print(f'Center [x, y] = [{x_center}, {y_center}]')

        history = f'[x, y] = [{round(x_center[j], 2)}, {round(y_center[j], 2)}]'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('WaffleCenteringModule', history)
        self.m_image_out_port.close_port()
