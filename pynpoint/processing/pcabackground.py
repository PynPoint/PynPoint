"""
Pipeline modules for PCA-based background subtraction.
"""

import math
import time
import warnings

from typing import List, Optional, Tuple, Union

import numpy as np

from scipy.optimize import curve_fit
from scipy.sparse.linalg import svds
from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.processing.psfpreparation import SortParangModule
from pynpoint.processing.resizing import CropImagesModule
from pynpoint.processing.stacksubset import CombineTagsModule
from pynpoint.util.module import memory_frames, progress
from pynpoint.util.star import locate_star


class PCABackgroundPreparationModule(ProcessingModule):
    """
    Pipeline module for preparing the images for a PCA-based background subtraction.
    """

    __author__ = 'Tomas Stolker, Silvan Hunziker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 star_out_tag: str,
                 subtracted_out_tag: str,
                 background_out_tag: str,
                 dither: Union[Tuple[int, int, int],
                               Tuple[int, None, Tuple[float, float]]],
                 combine: str = 'mean') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the pipeline module instance.
        image_in_tag : str
            Database tag with the images that are read as input.
        star_out_tag : str
            Database tag to store the images that contain the star.
        subtracted_out_tag : str
            Database tag to store the mean/median background subtracted images with the star.
        background_out_tag : str
            Database tag to store the images that contain only background emission.
        dither : tuple(int, int, int), tuple(int, None, tuple(float, float))
            Tuple with the parameters for separating the star and background frames. The tuple
            should contain three values, ``(positions, cubes, first)``, with ``positions`` the
            number of unique dithering position, ``cubes`` the number of consecutive cubes per
            dithering position, and ``first`` the index value of the first cube which contains the
            star (Python indexing starts at zero). Sorting is based on the ``DITHER_X`` and
            ``DITHER_Y`` attributes when ``cubes`` is set to None. In that case, the ``first``
            value should be a tuple with the ``DITHER_X`` and ``DITHER_Y`` values in which the star
            appears first.
        combine : str
            Method for combining the background images ('mean' or 'median').

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_star_out_port = self.add_output_port(star_out_tag)
        self.m_subtracted_out_port = self.add_output_port(subtracted_out_tag)
        self.m_background_out_port = self.add_output_port(background_out_tag)

        if len(dither) != 3:
            raise ValueError('The \'dither\' argument should contain three values.')

        self.m_dither = dither
        self.m_combine = combine

    @typechecked
    def _prepare(self) -> Tuple[np.ndarray, np.ndarray]:

        nframes = self.m_image_in_port.get_attribute('NFRAMES')

        cube_mean = np.zeros((nframes.shape[0],
                              self.m_image_in_port.get_shape()[2],
                              self.m_image_in_port.get_shape()[1]))

        count = 0
        for i, item in enumerate(nframes):
            if self.m_combine == 'mean':
                cube_mean[i, ] = np.mean(self.m_image_in_port[count:count+item, ], axis=0)

            elif self.m_combine == 'median':
                cube_mean[i, ] = np.median(self.m_image_in_port[count:count+item, ], axis=0)

            count += item

        if self.m_dither[1] is None:
            dither_x = self.m_image_in_port.get_attribute('DITHER_X')
            dither_y = self.m_image_in_port.get_attribute('DITHER_Y')

            star = np.logical_and(dither_x == self.m_dither[2][0],
                                  dither_y == self.m_dither[2][1])

            bg_frames = np.invert(star)

        else:
            bg_frames = np.ones(nframes.shape[0], dtype=bool)

            for i in range(self.m_dither[2]*self.m_dither[1],
                           np.size(nframes),
                           self.m_dither[1]*self.m_dither[0]):

                bg_frames[i:i+self.m_dither[1]] = False

        return bg_frames, cube_mean

    @typechecked
    def _separate(self,
                  bg_frames: np.ndarray,
                  bg_indices: np.ndarray,
                  parang: Optional[np.ndarray],
                  cube_mean: np.ndarray) -> Tuple[np.array, Optional[np.ndarray], np.ndarray,
                                                  np.ndarray, Optional[np.ndarray], np.ndarray]:

        @typechecked
        def _initialize() -> Tuple[np.array, Optional[np.ndarray], np.ndarray, np.ndarray,
                                   Optional[np.ndarray], np.ndarray]:

            background_nframes = np.empty(0, dtype=np.int64)
            star_nframes = np.empty(0, dtype=np.int64)

            background_index = np.empty(0, dtype=np.int64)
            star_index = np.empty(0, dtype=np.int64)

            if parang is None:
                background_parang = None
                star_parang = None

            else:
                background_parang = np.empty(0, dtype=np.float64)
                star_parang = np.empty(0, dtype=np.float64)

            return star_index, star_parang, star_nframes, background_index, \
                background_parang, background_nframes

        @typechecked
        def _select_background(i: int) -> np.ndarray:

            # Previous background cube
            if np.size(bg_indices[bg_indices < i]) > 0:
                index_prev = np.amax(bg_indices[bg_indices < i])
                bg_prev = cube_mean[index_prev, ]

            else:
                bg_prev = None

            # Next background cube
            if np.size(bg_indices[bg_indices > i]) > 0:
                index_next = np.amin(bg_indices[bg_indices > i])
                bg_next = cube_mean[index_next, ]

            else:
                bg_next = None

            # Select background: previous, next, or mean of previous and next
            if bg_prev is None and bg_next is not None:
                background = bg_next

            elif bg_prev is not None and bg_next is None:
                background = bg_prev

            elif bg_prev is not None and bg_next is not None:
                background = (bg_prev+bg_next)/2.

            else:
                raise ValueError('Neither previous nor next background frames are found.')

            return background

        nframes = self.m_image_in_port.get_attribute('NFRAMES')
        index = self.m_image_in_port.get_attribute('INDEX')

        star_index, star_parang, star_nframes, background_index, background_parang, \
            background_nframes = _initialize()

        # Separate star and background cubes. Subtract mean background.
        count = 0

        start_time = time.time()
        for i, item in enumerate(nframes):
            progress(i, len(nframes), 'Preparing PCA background subtraction...', start_time)

            im_tmp = self.m_image_in_port[count:count+item, ]

            # Background frames
            if bg_frames[i]:

                self.m_background_out_port.append(im_tmp)

                background_nframes = np.append(background_nframes, nframes[i])
                background_index = np.append(background_index, index[count:count+item])

                if parang is not None:
                    background_parang = np.append(background_parang, parang[count:count+item])

            # Star frames
            else:
                background = _select_background(i)

                self.m_star_out_port.append(im_tmp)
                self.m_subtracted_out_port.append(im_tmp-background)

                star_nframes = np.append(star_nframes, nframes[i])
                star_index = np.append(star_index, index[count:count+item])

                if parang is not None:
                    star_parang = np.append(star_parang, parang[count:count+item])

            count += item

        return star_index, star_parang, star_nframes, background_index, \
            background_parang, background_nframes

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Separates the star and background frames, subtracts the mean
        or median background from both the star and background frames, and writes the star and
        background frames separately to their respective output ports.

        Returns
        -------
        NoneType
            None
        """

        if 'PARANG' in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute('PARANG')
        else:
            parang = None

        bg_frames, cube_mean = self._prepare()

        bg_indices = np.nonzero(bg_frames)[0]

        star_index, star_parang, star_nframes, background_index, background_parang, \
            background_nframes = self._separate(bg_frames, bg_indices, parang, cube_mean)

        history = f'frames = {sum(star_nframes)}, {len(background_nframes)}'

        self.m_star_out_port.copy_attributes(self.m_image_in_port)
        self.m_star_out_port.add_history('PCABackgroundPreparationModule', history)
        self.m_star_out_port.add_attribute('NFRAMES', star_nframes, static=False)
        self.m_star_out_port.add_attribute('INDEX', star_index, static=False)

        if parang is not None:
            self.m_star_out_port.add_attribute('PARANG', star_parang, static=False)

        self.m_subtracted_out_port.copy_attributes(self.m_image_in_port)
        self.m_subtracted_out_port.add_history('PCABackgroundPreparationModule', history)
        self.m_subtracted_out_port.add_attribute('NFRAMES', star_nframes, static=False)
        self.m_subtracted_out_port.add_attribute('INDEX', star_index, static=False)

        if parang is not None:
            self.m_subtracted_out_port.add_attribute('PARANG', star_parang, static=False)

        self.m_background_out_port.copy_attributes(self.m_image_in_port)
        self.m_background_out_port.add_history('PCABackgroundPreparationModule', history)
        self.m_background_out_port.add_attribute('NFRAMES', background_nframes, static=False)
        self.m_background_out_port.add_attribute('INDEX', background_index, static=False)

        if parang is not None:
            self.m_background_out_port.add_attribute('PARANG', background_parang, static=False)

        self.m_star_out_port.close_port()


class PCABackgroundSubtractionModule(ProcessingModule):
    """
    Pipeline module applying a PCA-based background subtraction (see Hunziker et al. 2018).
    """

    __author__ = 'Tomas Stolker, Silvan Hunziker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 star_in_tag: str,
                 background_in_tag: str,
                 residuals_out_tag: str,
                 fit_out_tag: Optional[str] = None,
                 mask_out_tag: Optional[str] = None,
                 pca_number: int = 60,
                 mask_star: float = 0.7,
                 subframe: Optional[float] = None,
                 gaussian: float = 0.15,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the pipeline module instance.
        star_in_tag : str
            Database tag with the input images that contain the star.
        background_in_tag : str
            Database tag with the input images that contain only background emission.
        residuals_out_tag : str
            Database tag to store the background-subtracted images of the star.
        fit_out_tag : str, None
            Database tag to store the modeled background images. The data is not stored if the
            arguments is set to None.
        mask_out_tag : str, None
            Database tag to store the mask. The data is not stored if the argument is set to None.
        pca_number : int
            Number of principal components that is used to model the background emission.
        mask_star : float
            Radius of the central mask (arcsec).
        gaussian : float
            Full width at half maximum (arcsec) of the Gaussian kernel that is used to smooth the
            image before the star is located.
        subframe : float, None
            Size (arcsec) of the subframe that is used to find the star. Cropping of the subframe
            is done around the center of the image. The full images is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        if 'subtract_mean' in kwargs:
            warnings.warn('The \'subtract_mean\' parameter has been deprecated. Subtracting of '
                          'the mean is no longer optional so subtract_mean=True.',
                          DeprecationWarning)

        self.m_star_in_port = self.add_input_port(star_in_tag)
        self.m_background_in_port = self.add_input_port(background_in_tag)
        self.m_residuals_out_port = self.add_output_port(residuals_out_tag)

        if fit_out_tag is None:
            self.m_fit_out_port = None
        else:
            self.m_fit_out_port = self.add_output_port(fit_out_tag)

        if mask_out_tag is None:
            self.m_mask_out_port = None
        else:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)

        self.m_pca_number = pca_number
        self.m_mask_star = mask_star
        self.m_gaussian = gaussian
        self.m_subframe = subframe

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Creates a PCA basis set of the background frames after
        subtracting the mean background frame from both the star and background frames, masks the
        PSF of the star, projects the star frames onto the principal components, and stores the
        residuals of the background subtracted images.

        Returns
        -------
        NoneType
            None
        """

        @typechecked
        def _create_mask(radius: float,
                         position: np.ndarray,
                         nimages: int) -> np.ndarray:
            """
            Method for creating a circular mask at the star or planet position.
            """

            npix = self.m_star_in_port.get_shape()[1]

            x_grid = np.arange(0, npix, 1)
            y_grid = np.arange(0, npix, 1)

            xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

            mask = np.ones((nimages, npix, npix))

            cent_x = position[:, 1]
            cent_y = position[:, 0]

            for i in range(nimages):
                rr_grid = np.sqrt((xx_grid - cent_x[i])**2 + (yy_grid - cent_y[i])**2)
                mask[i, ][rr_grid < radius] = 0.

            return mask

        @typechecked
        def _create_basis(images: np.ndarray,
                          pca_number: int) -> np.ndarray:
            """
            Method for calculating the principal components for a stack of background images.

            Parameters
            ----------
            images : np.ndarray
                Background images with the mean subtracted from all images.
            pca_number : int
                Number of principal components that is used to model the background emission.

            Returns
            -------
            np.ndarray
                Principal components with the second and third dimension reshaped to ``images``.
            """

            _, _, v_svd = svds(images.reshape(images.shape[0],
                                              images.shape[1]*images.shape[2]),
                               k=pca_number)

            v_svd = v_svd[::-1, ]

            return v_svd.reshape(v_svd.shape[0], images.shape[1], images.shape[2])

        @typechecked
        def _model_background(basis: np.ndarray,
                              im_arr: np.ndarray,
                              mask: np.ndarray) -> np.ndarray:
            """
            Method for creating a model of the background.
            """

            @typechecked
            def _dot_product(x_dot: np.ndarray,
                             *p: np.float64) -> np.ndarray:
                return np.dot(p, x_dot)

            fit_im_chi = np.zeros(im_arr.shape)
            # fit_coeff_chi = np.zeros((im_arr.shape[0], basis.shape[0]))

            basis_reshaped = basis.reshape(basis.shape[0], -1)

            for i in range(im_arr.shape[0]):
                basis_reshaped_masked = (basis*mask[i]).reshape(basis.shape[0], -1)

                data_to_fit = im_arr[i, ]

                init = np.ones(basis_reshaped_masked.shape[0])

                fitted = curve_fit(_dot_product,
                                   basis_reshaped_masked,
                                   data_to_fit.reshape(-1),
                                   init)

                fit_im = np.dot(fitted[0], basis_reshaped)
                fit_im = fit_im.reshape(data_to_fit.shape[0], data_to_fit.shape[1])

                fit_im_chi[i, ] = fit_im
                # fit_coeff_chi[i, ] = fitted[0]

            return fit_im_chi

        memory = self._m_config_port.get_attribute('MEMORY')
        pixscale = self.m_star_in_port.get_attribute('PIXSCALE')

        nimages = self.m_star_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        self.m_mask_star /= pixscale
        self.m_gaussian = int(math.ceil(self.m_gaussian/pixscale))

        if self.m_subframe is not None:
            self.m_subframe /= pixscale
            self.m_subframe = int(math.ceil(self.m_subframe))

        bg_mean = np.mean(self.m_background_in_port.get_all(), axis=0)

        star = np.zeros((nimages, 2))
        for i, _ in enumerate(star):
            star[i, :] = locate_star(image=self.m_star_in_port[i, ] - bg_mean,
                                     center=None,
                                     width=self.m_subframe,
                                     fwhm=self.m_gaussian)

        print('Creating PCA basis set...', end='')

        basis_pca = _create_basis(self.m_background_in_port.get_all() - bg_mean,
                                  self.m_pca_number)

        print(' [DONE]')

        start_time = time.time()
        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Calculating background model...', start_time)

            # Subtract the mean background from the star frames
            im_star = self.m_star_in_port[frames[i]:frames[i+1], ] - bg_mean

            mask = _create_mask(self.m_mask_star,
                                star[frames[i]:frames[i+1], ],
                                int(frames[i+1]-frames[i]))

            fit_im = _model_background(basis_pca, im_star*mask, mask)

            self.m_residuals_out_port.append(im_star-fit_im)

            if self.m_fit_out_port is not None:
                self.m_fit_out_port.append(fit_im)

            if self.m_mask_out_port is not None:
                self.m_mask_out_port.append(mask)

        history = f'PC number = {self.m_pca_number}'
        self.m_residuals_out_port.copy_attributes(self.m_star_in_port)
        self.m_residuals_out_port.add_history('PCABackgroundSubtractionModule', history)
        self.m_residuals_out_port.add_attribute('STAR_POSITION', star, static=False)

        if self.m_fit_out_port is not None:
            self.m_fit_out_port.copy_attributes(self.m_star_in_port)
            self.m_fit_out_port.add_history('PCABackgroundSubtractionModule', history)

        if self.m_mask_out_port is not None:
            self.m_mask_out_port.copy_attributes(self.m_star_in_port)
            self.m_mask_out_port.add_history('PCABackgroundSubtractionModule', history)

        self.m_residuals_out_port.close_port()


class DitheringBackgroundModule(ProcessingModule):
    """
    Pipeline module for PCA-based background subtraction of dithering data. This is a wrapper that
    applies the processing modules for either a mean or the PCA-based background subtraction.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 center: Optional[List[Tuple[int, int]]] = None,
                 cubes: Optional[int] = None,
                 size: float = 2.,
                 gaussian: float = 0.15,
                 subframe: Optional[float] = None,
                 pca_number: Optional[int] = 5,
                 mask_star: float = 0.7,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Database tag with input images.
        image_out_tag : str
            Database tag to store the background subtracted images.
        center : list(tuple(int, int)), None
            Tuple with the centers of the dithering positions, e.g. ((x0, y0), (x1, y1)). The order
            of the coordinates should correspond to the order in which the star is present. If
            ``center`` and ``cubes`` are both set to None then sorting and subtracting of the
            background frames is based on ``DITHER_X`` and ``DITHER_Y``. If ``center`` is
            specified and ``cubes`` is set to None then the ``DITHER_X`` and ``DITHER_Y``
            attributes will be used for sorting and subtracting of the background but not for
            selecting the dither positions.
        cubes : int, None
            Number of consecutive cubes per dither position. If ``cubes`` is set to None then
            sorting and subtracting of the background frames is based on ``DITHER_X`` and
            ``DITHER_Y``.
        size : float
            Cropped image size (arcsec).
        gaussian : float
            Full width at half maximum (arcsec) of the Gaussian kernel that is used to smooth the
            image before the star is located.
        subframe : float, None
            Size (arcsec) of the subframe that is used to search for the star. Cropping of the
            subframe is done around the center of the dithering position. The full image size
            (i.e. ``size``) will be used if set to None then.
        pca_number : int, None
            Number of principal components that is used to model the background emission. The PCA
            background subtraction is skipped if the argument is set to None. In that case, the
            mean background subtracted images are written toe ``image_out_tag``.
        mask_star : float
            Radius of the central mask (arcsec) that is used to exclude the star when fitting the
            principal components. The region behind the mask is included when subtracting the
            PCA background model.

        Returns
        -------
        NoneType
            None
        """

        if 'mask_planet' in kwargs:
            warnings.warn('The \'mask_planet\' parameter has been deprecated.', DeprecationWarning)

        if 'crop' in kwargs:
            warnings.warn('The \'crop\' parameter has been deprecated. The step to crop the '
                          'images is no longer optional so crop=True.', DeprecationWarning)

        if 'prepare' in kwargs:
            warnings.warn('The \'prepare\' parameter has been deprecated. The preparation step '
                          'is no longer optional so prepare=True.', DeprecationWarning)

        if 'pca_background' in kwargs:
            warnings.warn('The \'pca_background\' parameter has been deprecated. The PCA '
                          'background is no longer optional when combine=\'pca\' so '
                          'pca_background=True.', DeprecationWarning)

        if 'subtract_mean' in kwargs:
            warnings.warn('The \'subtract_mean\' parameter has been deprecated. Subtracting of '
                          'the mean is no longer optional so subtract_mean=True.',
                          DeprecationWarning)

        if 'combine' in kwargs:
            warnings.warn('The \'combine\' parameter has been deprecated. To write the mean '
                          'background subtracted images to image_out_tag is done by setting '
                          'pca_number=None.', DeprecationWarning)

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_center = center
        self.m_cubes = cubes
        self.m_size = size
        self.m_gaussian = gaussian
        self.m_pca_number = pca_number
        self.m_mask_star = mask_star
        self.m_subframe = subframe

        self.m_image_in_tag = image_in_tag
        self.m_image_out_tag = image_out_tag

    @typechecked
    def _initialize(self) -> Tuple[int, np.ndarray]:
        if self.m_cubes is None:
            dither_x = self.m_image_in_port.get_attribute('DITHER_X')
            dither_y = self.m_image_in_port.get_attribute('DITHER_Y')

            dither_xy = np.zeros((dither_x.shape[0], 2))
            dither_xy[:, 0] = dither_x
            dither_xy[:, 1] = dither_y

            _, index = np.unique(dither_xy, axis=0, return_index=True)

            dither = dither_xy[np.sort(index)]

            npix = self.m_image_in_port.get_shape()[1]

            # Compute center from dither and make sure all positions are actually Python integers
            if self.m_center is None:
                self.m_center = np.copy(dither) + float(npix) / 2.

                self.m_center = tuple(zip(map(int, self.m_center[:, 0]),
                                          map(int, self.m_center[:, 1])))

            else:
                if np.size(dither, axis=0) != np.size(self.m_center, axis=0):
                    raise ValueError('Number of specified center positions should be equal to the '
                                     'number of unique dithering positions.')

        n_dither = np.size(self.m_center, 0)

        if self.m_cubes is None:
            star_pos = np.copy(dither)
        else:
            star_pos = np.arange(0, n_dither, 1)

        return n_dither, star_pos

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Cuts out the detector sections at the different dither positions,
        prepares the PCA background subtraction, locates the star in each image, runs the PCA
        background subtraction, combines the output from the different dither positions is written
        to a single database tag.

        Returns
        -------
        NoneType
            None
        """

        @typechecked
        def _admin_start(count: int,
                         n_dither: int,
                         position: Tuple[int, int],
                         star_pos: Union[np.ndarray, np.int64]) -> None:
            print(f'Processing dither position {count+1} out of {n_dither}...')
            print(f'Center position = {position}')

            if self.m_cubes is None and self.m_center is not None:
                print(f'DITHER_X, DITHER_Y = {tuple(star_pos)}')

        @typechecked
        def _admin_end(count: int) -> None:
            if self.m_pca_number is None:
                tags.append(f'{self.m_image_in_tag}_dither_mean{count+1}')

            else:
                tags.append(f'{self.m_image_in_tag}_dither_pca_res{count+1}')

        n_dither, star_pos = self._initialize()
        tags = []

        for i, position in enumerate(self.m_center):
            _admin_start(i, n_dither, position, star_pos[i])

            im_out_tag = f'{self.m_image_in_tag}_dither_crop{i+1}'

            module = CropImagesModule(name_in=f'crop{i}',
                                      image_in_tag=self.m_image_in_tag,
                                      image_out_tag=im_out_tag,
                                      size=self.m_size,
                                      center=(int(math.ceil(position[0])),
                                              int(math.ceil(position[1]))))

            module.connect_database(self._m_data_base)
            module._m_output_ports[im_out_tag].del_all_data()
            module._m_output_ports[im_out_tag].del_all_attributes()
            module.run()

            if self.m_cubes is None:
                dither_val = (n_dither, self.m_cubes, tuple(star_pos[i]))
            else:
                dither_val = (n_dither, self.m_cubes, int(star_pos[i]))

            im_in_tag = f'{self.m_image_in_tag}_dither_crop{i+1}'
            star_out_tag = f'{self.m_image_in_tag}_dither_star{i+1}'
            sub_out_tag = f'{self.m_image_in_tag}_dither_mean{i+1}'
            back_out_tag = f'{self.m_image_in_tag}_dither_background{i+1}'

            module = PCABackgroundPreparationModule(name_in=f'prepare{i}',
                                                    image_in_tag=im_in_tag,
                                                    star_out_tag=star_out_tag,
                                                    subtracted_out_tag=sub_out_tag,
                                                    background_out_tag=back_out_tag,
                                                    dither=dither_val,
                                                    combine='mean')

            module.connect_database(self._m_data_base)
            module._m_output_ports[star_out_tag].del_all_data()
            module._m_output_ports[star_out_tag].del_all_attributes()
            module._m_output_ports[sub_out_tag].del_all_data()
            module._m_output_ports[sub_out_tag].del_all_attributes()
            module._m_output_ports[back_out_tag].del_all_data()
            module._m_output_ports[back_out_tag].del_all_attributes()
            module.run()

            if self.m_pca_number is not None:
                star_in_tag = f'{self.m_image_in_tag}_dither_star{i+1}'
                back_in_tag = f'{self.m_image_in_tag}_dither_background{i+1}'
                res_out_tag = f'{self.m_image_in_tag}_dither_pca_res{i+1}'
                fit_out_tag = f'{self.m_image_in_tag}_dither_pca_fit{i+1}'
                mask_out_tag = f'{self.m_image_in_tag}_dither_pca_mask{i+1}'

                module = PCABackgroundSubtractionModule(name_in=f'pca_background{i}',
                                                        star_in_tag=star_in_tag,
                                                        background_in_tag=back_in_tag,
                                                        residuals_out_tag=res_out_tag,
                                                        fit_out_tag=fit_out_tag,
                                                        mask_out_tag=mask_out_tag,
                                                        pca_number=self.m_pca_number,
                                                        mask_star=self.m_mask_star,
                                                        subframe=self.m_subframe,
                                                        gaussian=self.m_gaussian)

                module.connect_database(self._m_data_base)
                module._m_output_ports[res_out_tag].del_all_data()
                module._m_output_ports[res_out_tag].del_all_attributes()
                module._m_output_ports[fit_out_tag].del_all_data()
                module._m_output_ports[fit_out_tag].del_all_attributes()
                module._m_output_ports[mask_out_tag].del_all_data()
                module._m_output_ports[mask_out_tag].del_all_attributes()
                module.run()

            _admin_end(i)

        module = CombineTagsModule(name_in='combine',
                                   check_attr=True,
                                   index_init=False,
                                   image_in_tags=tags,
                                   image_out_tag=self.m_image_in_tag+'_dither_combine')

        module.connect_database(self._m_data_base)
        module._m_output_ports[self.m_image_in_tag+'_dither_combine'].del_all_data()
        module._m_output_ports[self.m_image_in_tag+'_dither_combine'].del_all_attributes()
        module.run()

        module = SortParangModule(name_in='sort',
                                  image_in_tag=self.m_image_in_tag+'_dither_combine',
                                  image_out_tag=self.m_image_out_tag)

        module.connect_database(self._m_data_base)
        module._m_output_ports[self.m_image_out_tag].del_all_data()
        module._m_output_ports[self.m_image_out_tag].del_all_attributes()
        module.run()
