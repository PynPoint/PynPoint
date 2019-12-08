"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings
import multiprocessing as mp

from typing import Union, Tuple

import numpy as np

from typeguard import typechecked
from skimage.metrics import structural_similarity, mean_squared_error

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import crop_image, pixel_distance, center_pixel
from pynpoint.util.module import progress
from pynpoint.util.remove import write_selected_data, write_selected_attributes
from pynpoint.util.star import star_positions


class RemoveFramesModule(ProcessingModule):
    """
    Pipeline module for removing images by their index number.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 selected_out_tag: str,
                 removed_out_tag: str,
                 frames: Union[str, range, list, np.ndarray]) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        selected_out_tag : str
            Tag of the database entry with the remaining images after removing the specified
            images. Should be different from *image_in_tag*.
        removed_out_tag : str
            Tag of the database entry with the images that are removed. Should be different
            from *image_in_tag*.
        frames : str, list, range, numpy.ndarray
            The frame indices that have to be removed or a database tag pointing to a list of
            frame indices.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        self.m_selected_out_port = self.add_output_port(selected_out_tag)
        self.m_removed_out_port = self.add_output_port(removed_out_tag)

        if isinstance(frames, str):
            self.m_index_in_port = self.add_input_port(frames)

        else:
            self.m_index_in_port = None

            if isinstance(frames, (range, list)):
                self.m_frames = np.asarray(frames, dtype=np.int)

            elif isinstance(frames, np.ndarray):
                self.m_frames = frames

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Removes the frames and corresponding attributes, updates the
        NFRAMES attribute, and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        self.m_selected_out_port.del_all_data()
        self.m_selected_out_port.del_all_attributes()

        self.m_removed_out_port.del_all_data()
        self.m_removed_out_port.del_all_attributes()

        if self.m_index_in_port is not None:
            self.m_frames = self.m_index_in_port.get_all()

        if np.size(np.where(self.m_frames >= self.m_image_in_port.get_shape()[0])) > 0:
            raise ValueError(f'Some values in \'frames\' are larger than the total number of '
                             f'available frames, {self.m_image_in_port.get_shape()[0]}.')

        write_selected_data(memory=self._m_config_port.get_attribute('MEMORY'),
                            indices=self.m_frames,
                            image_in_port=self.m_image_in_port,
                            selected_out_port=self.m_selected_out_port,
                            removed_out_port=self.m_removed_out_port)

        write_selected_attributes(indices=self.m_frames,
                                  image_in_port=self.m_image_in_port,
                                  selected_out_port=self.m_selected_out_port,
                                  removed_out_port=self.m_removed_out_port,
                                  module_type='RemoveFramesModule',
                                  history=f'frames removed = {np.size(self.m_frames)}')

        self.m_image_in_port.close_port()


class FrameSelectionModule(ProcessingModule):
    """
    Pipeline module for applying a frame selection.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 selected_out_tag: str,
                 removed_out_tag: str,
                 index_out_tag: str = None,
                 method: str = 'median',
                 threshold: float = 4.,
                 fwhm: float = 0.1,
                 aperture: Union[Tuple[str, float], Tuple[str, float, float]] = ('circular', 0.2),
                 position: Union[Tuple[int, int, float], Tuple[None, None, float],
                                 Tuple[int, int, None]] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        selected_out_tag : str
            Tag of the database entry with the selected images that are written as output. Should
            be different from `image_in_tag`.
        removed_out_tag : str
            Tag of the database entry with the removed images that are written as output. Should
            be different from `image_in_tag`.
        index_out_tag : str, None
            Tag of the database entry with the list of frames indices that are removed with the
            frames selection. No data is written when set to None.
        method : str
            Perform the sigma clipping with respect to the median or maximum aperture flux by
            setting the `method` to 'median' or 'max'.
        threshold : float
            Threshold in units of sigma for the frame selection. All images that are a `threshold`
            number of sigmas away from the median/maximum photometry will be removed.
        fwhm : float
            The FWHM (arcsec) of the Gaussian kernel that is used to smooth the images before the
            brightest pixel is located. Recommended to be similar in size to the FWHM of the
            stellar PSF.
        aperture : tuple(str, float), tuple(str, float, float)
            Tuple with the aperture properties for measuring the photometry around the location of
            the brightest pixel. The first element contains the aperture type ('circular',
            'annulus', or 'ratio'). For a circular aperture, the second element contains the
            aperture radius (arcsec). For the other two types, the second and third element are the
            inner and outer radii (arcsec) of the aperture.
        position : tuple(int, int, float), None
            Subframe that is selected to search for the star. The tuple contains the center (pix)
            and size (arcsec) (pos_x, pos_y, size). Setting `position` to None will use the full
            image to search for the star. If `position=(None, None, size)` then the center of the
            image will be used. If `position=(pos_x, pos_y, None)` then a fixed position is used
            for the aperture.

        Returns
        -------
        NoneType
            None
        """

        super(FrameSelectionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        self.m_selected_out_port = self.add_output_port(selected_out_tag)
        self.m_removed_out_port = self.add_output_port(removed_out_tag)

        if index_out_tag is None:
            self.m_index_out_port = None
        else:
            self.m_index_out_port = self.add_output_port(index_out_tag)

        self.m_method = method
        self.m_fwhm = fwhm
        self.m_aperture = aperture
        self.m_threshold = threshold
        self.m_position = position

    @staticmethod
    @typechecked
    def aperture_phot(image: np.ndarray,
                      position: np.ndarray,
                      aperture: Union[Tuple[str, float, float],
                                      Tuple[str, None, float]]) -> np.float64:
        """
        Parameters
        ----------
        image : np.ndarray
            Input image (2D).
        position : np.ndarray
            Center position (y, x) of the aperture.
        aperture : tuple(str, float, float)
            Tuple with the aperture properties for measuring the photometry around the location of
            the brightest pixel. The first element contains the aperture type ('circular',
            'annulus', or 'ratio'). For a circular aperture, the second element is empty and the
            third element contains the aperture radius (pix). For the other two types, the
            second and third element are the inner and outer radii (pix) of the aperture.

        Returns
        -------
        np.float64
            Photometry value.
        """

        check_pos_in = any(np.floor(position[:]-aperture[2]) < 0.)
        check_pos_out = any(np.ceil(position[:]+aperture[2]) > image.shape[0])

        if check_pos_in or check_pos_out:
            phot = np.nan

        else:
            im_crop = crop_image(image, tuple(position), 2*int(math.ceil(aperture[2])))

            npix = im_crop.shape[0]

            x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)
            xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
            rr_grid = np.sqrt(xx_grid**2 + yy_grid**2)

            if aperture[0] == 'circular':
                phot = np.sum(im_crop[rr_grid < aperture[2]])

            elif aperture[0] == 'annulus':
                phot = np.sum(im_crop[(rr_grid > aperture[1]) & (rr_grid < aperture[2])])

            elif aperture[0] == 'ratio':
                phot = np.sum(im_crop[rr_grid < aperture[1]]) / \
                       np.sum(im_crop[(rr_grid > aperture[1]) & (rr_grid < aperture[2])])

        return phot

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Smooths the images with a Gaussian kernel, locates the brightest
        pixel in each image, measures the integrated flux around the brightest pixel, calculates
        the median and standard deviation of the photometry, and applies sigma clipping to remove
        low quality images.

        Returns
        -------
        NoneType
            None
        """

        self.m_selected_out_port.del_all_data()
        self.m_selected_out_port.del_all_attributes()

        self.m_removed_out_port.del_all_data()
        self.m_removed_out_port.del_all_attributes()

        if self.m_index_out_port is not None:
            self.m_index_out_port.del_all_data()
            self.m_index_out_port.del_all_attributes()

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        nimages = self.m_image_in_port.get_shape()[0]

        self.m_fwhm = int(math.ceil(self.m_fwhm/pixscale))

        if self.m_position is not None and self.m_position[2] is not None:
            self.m_position = (self.m_position[0],
                               self.m_position[0],
                               int(math.ceil(self.m_position[2]/pixscale)))

        if len(self.m_aperture) == 2:
            self.m_aperture = (self.m_aperture[0],
                               None,
                               self.m_aperture[1]/pixscale)

        elif len(self.m_aperture) == 3:
            self.m_aperture = (self.m_aperture[0],
                               self.m_aperture[1]/pixscale,
                               self.m_aperture[2]/pixscale)

        starpos = star_positions(self.m_image_in_port, self.m_fwhm, self.m_position)

        phot = np.zeros(nimages)
        start_time = time.time()

        for i in range(nimages):
            progress(i, nimages, 'Aperture photometry...', start_time)

            phot[i] = self.aperture_phot(self.m_image_in_port[i, ], starpos[i, :], self.m_aperture)

        if self.m_method == 'median':
            phot_ref = np.nanmedian(phot)
            print(f'Median = {phot_ref:.2f}')

        elif self.m_method == 'max':
            phot_ref = np.nanmax(phot)
            print(f'Maximum = {phot_ref:.2f}')

        phot_std = np.nanstd(phot)
        print(f'Standard deviation = {phot_std:.2f}')

        index_del = np.logical_or((phot > phot_ref + self.m_threshold*phot_std),
                                  (phot < phot_ref - self.m_threshold*phot_std))

        index_del[np.isnan(phot)] = True
        index_del = np.where(index_del)[0]

        index_sel = np.ones(nimages, dtype=bool)
        index_sel[index_del] = False

        write_selected_data(memory=self._m_config_port.get_attribute('MEMORY'),
                            indices=index_del,
                            image_in_port=self.m_image_in_port,
                            selected_out_port=self.m_selected_out_port,
                            removed_out_port=self.m_removed_out_port)

        history = f'frames removed = {np.size(index_del)}'

        if self.m_index_out_port is not None:
            self.m_index_out_port.set_all(index_del, data_dim=1)
            self.m_index_out_port.copy_attributes(self.m_image_in_port)
            self.m_index_out_port.add_attribute('STAR_POSITION', starpos, static=False)
            self.m_index_out_port.add_history('FrameSelectionModule', history)

        write_selected_attributes(indices=index_del,
                                  image_in_port=self.m_image_in_port,
                                  selected_out_port=self.m_selected_out_port,
                                  removed_out_port=self.m_removed_out_port,
                                  module_type='FrameSelectionModule',
                                  history=history)

        self.m_selected_out_port.add_attribute('STAR_POSITION', starpos[index_sel], static=False)
        self.m_selected_out_port.add_history('FrameSelectionModule', history)

        self.m_removed_out_port.add_attribute('STAR_POSITION', starpos[index_del], static=False)
        self.m_removed_out_port.add_history('FrameSelectionModule', history)

        self.m_image_in_port.close_port()


class RemoveLastFrameModule(ProcessingModule):
    """
    Pipeline module for removing every NDIT+1 image from NACO dataset obtained in cube mode. This
    frame contains the average pixel values of the cube.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveLastFrameModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Removes every NDIT+1 frame and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        ndit = self.m_image_in_port.get_attribute('NDIT')
        nframes = self.m_image_in_port.get_attribute('NFRAMES')
        index = self.m_image_in_port.get_attribute('INDEX')

        nframes_new = []
        index_new = []

        start_time = time.time()
        for i, item in enumerate(ndit):
            progress(i, len(ndit), 'Removing the last image of each FITS cube...', start_time)

            if nframes[i] != item+1:
                warnings.warn(f'Number of frames ({nframes[i]}) is not equal to NDIT+1.')

            frame_start = np.sum(nframes[0:i])
            frame_end = np.sum(nframes[0:i+1]) - 1

            nframes_new.append(nframes[i]-1)
            index_new.extend(index[frame_start:frame_end])

            self.m_image_out_port.append(self.m_image_in_port[frame_start:frame_end, ])

        nframes_new = np.asarray(nframes_new, dtype=np.int)
        index_new = np.asarray(index_new, dtype=np.int)

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.add_attribute('NFRAMES', nframes_new, static=False)
        self.m_image_out_port.add_attribute('INDEX', index_new, static=False)

        history = 'frames removed = NDIT+1'
        self.m_image_out_port.add_history('RemoveLastFrameModule', history)

        self.m_image_out_port.close_port()


class RemoveStartFramesModule(ProcessingModule):
    """
    Pipeline module for removing a fixed number of images at the beginning of each cube. This can
    be useful for NACO data in which the background is higher at the beginning of the cube.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 frames: int = 1) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        frames : int
            Number of frames that are removed at the beginning of each cube.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveStartFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_frames = int(frames)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Removes a constant number of images at the beginning of each cube
        and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError('Input and output port should have a different tag.')

        nframes = self.m_image_in_port.get_attribute('NFRAMES')
        index = self.m_image_in_port.get_attribute('INDEX')

        index_new = []

        if 'PARANG' in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute('PARANG')
            parang_new = []

        else:
            parang = None

        if 'STAR_POSITION' in self.m_image_in_port.get_all_non_static_attributes():
            star = self.m_image_in_port.get_attribute('STAR_POSITION')
            star_new = []

        else:
            star = None

        start_time = time.time()
        for i, _ in enumerate(nframes):
            progress(i, len(nframes), 'Removing images at the begin of each cube...', start_time)

            frame_start = np.sum(nframes[0:i]) + self.m_frames
            frame_end = np.sum(nframes[0:i+1])

            if frame_start >= frame_end:
                raise ValueError('The number of frames in the original data cube is equal or '
                                 'smaller than the number of frames that have to be removed.')

            index_new.extend(index[frame_start:frame_end])

            if parang is not None:
                parang_new.extend(parang[frame_start:frame_end])

            if star is not None:
                star_new.extend(star[frame_start:frame_end])

            self.m_image_out_port.append(self.m_image_in_port[frame_start:frame_end, ])

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.add_attribute('NFRAMES', nframes-self.m_frames, static=False)
        self.m_image_out_port.add_attribute('INDEX', index_new, static=False)

        if parang is not None:
            self.m_image_out_port.add_attribute('PARANG', parang_new, static=False)

        if star is not None:
            self.m_image_out_port.add_attribute('STAR_POSITION', np.asarray(star_new), static=False)

        history = 'frames removed = '+str(self.m_frames)
        self.m_image_out_port.add_history('RemoveStartFramesModule', history)

        self.m_image_out_port.close_port()


class ImageStatisticsModule(ProcessingModule):
    """
    Pipeline module for calculating image statistics for the full images or a subsection of the
    images.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 stat_out_tag: str,
                 position: Union[Tuple[int, int, float], Tuple[None, None, float]] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the images that are read as input.
        stat_out_tag : str
            Tag of the database entry with the statistical results that are written as output. The
            result is stored in the following order: minimum, maximum, sum, mean, median, and
            standard deviation.
        position : tuple(int, int, float)
            Position (x, y) (pix) and radius (arcsec) of the circular area in which the statistics
            are calculated. The full image is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        super(ImageStatisticsModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_stat_out_port = self.add_output_port(stat_out_tag)

        self.m_position = position

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Calculates the minimum, maximum, sum, mean, median, and standard
        deviation of the pixel values of each image separately. NaNs are ignored for each
        calculation. The values are calculated for either the full images or a circular
        subsection of the images.

        Returns
        -------
        NoneType
            None
        """

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        nimages = self.m_image_in_port.get_shape()[0]
        im_shape = self.m_image_in_port.get_shape()[1:]

        if self.m_position is None:
            indices = None

        else:
            if self.m_position[0] is None and self.m_position[1] is None:
                center = center_pixel(self.m_image_in_port[0, ])

                self.m_position = (center[0],  # y position
                                   center[1],  # x position
                                   self.m_position[2]/pixscale)  # radius (pix)

            else:
                self.m_position = (int(self.m_position[1]),  # y position
                                   int(self.m_position[0]),  # x position
                                   self.m_position[2]/pixscale)  # radius (pix)

            rr_grid = pixel_distance(im_shape, self.m_position[0:2])
            rr_reshape = np.reshape(rr_grid, (rr_grid.shape[0]*rr_grid.shape[1]))
            indices = np.where(rr_reshape <= self.m_position[2])[0]

        def _image_stat(image_in, indices):
            if indices is None:
                image_select = np.copy(image_in)

            else:
                image_reshape = np.reshape(image_in, (image_in.shape[0]*image_in.shape[1]))
                image_select = image_reshape[indices]

            nmin = np.nanmin(image_select)
            nmax = np.nanmax(image_select)
            nsum = np.nansum(image_select)
            mean = np.nanmean(image_select)
            median = np.nanmedian(image_select)
            std = np.nanstd(image_select)

            return np.asarray([nmin, nmax, nsum, mean, median, std])

        self.apply_function_to_images(_image_stat,
                                      self.m_image_in_port,
                                      self.m_stat_out_port,
                                      'Calculating image statistics',
                                      func_args=(indices, ))

        history = f'number of images = {nimages}'
        self.m_stat_out_port.copy_attributes(self.m_image_in_port)
        self.m_stat_out_port.add_history('ImageStatisticsModule', history)
        self.m_stat_out_port.close_port()


class FrameSimilarityModule(ProcessingModule):
    """
    Pipeline module for measuring the similarity between frames.
    """

    __author__ = 'Benedikt Schmidhuber, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_tag: str,
                 method: str = 'MSE',
                 mask_radius: Tuple[float, float] = (0., 5.),
                 window_size: float = 0.1,
                 temporal_median: str = 'full') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_tag : str
            Tag of the database entry that is read as input.
        method : str
            Method for the similarity measure. There are three measures available:

                - `MSE` - Mean Squared Error
                - `PCC` - Pearson Correlation Coefficient
                - `SSIM` - Structural Similarity

            These measures compare each image to the temporal median of the image set.
        mask_radius : tuple(float, float)
            Inner and outer radius (arcsec) of the mask that is applied to the images.
        window_size : float
            Size (arcsec) of the sliding window that is used when the SSIM similarity is
            calculated.
        temporal_median : str
            Option to calculate the temporal median for each position ('full') or as a constant
            value ('constant') for the entire set. The latter is computationally less expensive.

        Returns
        -------
        NoneType
            None
        """

        super(FrameSimilarityModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_tag)
        self.m_image_out_port = self.add_output_port(image_tag)

        if method not in ('MSE', 'PCC', 'SSIM'):
            raise ValueError(f'The chosen method \'{method}\' is not available. Please ensure '
                             f'that you have selected one of \'MSE\', \'PCC\', or \'SSIM\'.')

        if temporal_median not in ('full', 'constant'):
            raise ValueError(f'The chosen temporal_median \'{temporal_median}\' is not '
                             f'available. Please ensure that you have selected one of \'full\', '
                             f'\'constant\'.')

        self.m_method = method
        self.m_temporal_median = temporal_median
        self.m_mask_radii = mask_radius
        self.m_window_size = window_size

    @staticmethod
    def _similarity(images, reference_index, mode, window_size, temporal_median=False):
        """
        Internal function to compute the MSE as defined by Ruane et al. 2019.
        """

        def _temporal_median(reference_index, images):
            """
            Internal function to calculate the temporal median for all frames, except the one with
            the ``reference_index``.
            """

            image_m = np.concatenate((images[:reference_index], images[reference_index+1:]))

            return np.median(image_m, axis=0)

        image_x_i = images[reference_index]

        if isinstance(temporal_median, bool):
            image_m = _temporal_median(reference_index, images=images)
        else:
            image_m = temporal_median

        if mode == 'MSE':
            return reference_index, mean_squared_error(image_x_i, image_m)

        if mode == 'PCC':
            # calculate the covariance matrix of the flattend images
            cov_mat = np.cov(image_x_i.flatten(), image_m.flatten(), ddof=1)

            # the variances are stored in the diagonal, therefore take the sqrt to obtain std
            std = np.sqrt(np.diag(cov_mat))

            # does not matter whether [0, 1] or [1, 0] as cov_mat is symmetrics
            return reference_index, cov_mat[0, 1] / (std[0] * std[1])

        if mode == 'SSIM':
            # winsize needs to be odd
            if int(window_size) % 2 == 0:
                winsize = int(window_size) + 1
            else:
                winsize = int(window_size)
            return reference_index, structural_similarity(image_x_i, image_m, win_size=winsize)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Computes the similarity between frames based on the Mean Squared
        Error (MSE), the Pearson Correlation Coefficient (PCC), or the Structural Similarity (SSIM).
        The correlation values are stored as non-static attribute (``MSE``, ``PCC``, or ``SSIM``)
        to the input data. After running this module, the
        :class:`~pynpoint.processing.frameselection.SelectByAttributeModule` can be used to select
        the images with the highest correlation.

        Returns
        -------
        NoneType
            None
        """

        cpu = self._m_config_port.get_attribute('CPU')
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        # get number of images
        nimages = self.m_image_in_port.get_shape()[0]

        # convert arcsecs to pixels
        self.m_mask_radii = (math.floor(self.m_mask_radii[0] / pixscale),
                             math.floor(self.m_mask_radii[1] / pixscale))
        self.m_window_size = int(self.m_window_size / pixscale)

        # overlay the same mask over all images
        images = self.m_image_in_port.get_all()

        # close the port during the calculations
        self.m_image_out_port.close_port()

        images = crop_image(images, None, int(self.m_mask_radii[1]))

        if self.m_temporal_median == 'constant':
            temporal_median = np.median(images, axis=0)
        else:
            temporal_median = False

        # compare images and store similarity
        similarities = np.zeros(nimages)

        pool = mp.Pool(cpu)
        async_results = []

        for i in range(nimages):
            async_results.append(pool.apply_async(FrameSimilarityModule._similarity,
                                                  args=(images,
                                                        i,
                                                        self.m_method,
                                                        self.m_window_size,
                                                        temporal_median)))

        pool.close()

        start_time = time.time()

        # wait for all processes to finish
        while mp.active_children():
            # number of finished processes
            nfinished = sum([i.ready() for i in async_results])

            progress(nfinished, nimages, 'Calculating image similarity...', start_time)

            # check if new processes have finished every 5 seconds
            time.sleep(5)

        if nfinished != nimages:
            sys.stdout.write('\r                                                      ')
            sys.stdout.write('\rCalculating image similarity... [DONE]\n')
            sys.stdout.flush()

        # get the results for every async_result object
        for async_result in async_results:
            reference, similarity = async_result.get()
            similarities[reference] = similarity

        pool.terminate()

        # reopen the port after the calculation
        self.m_image_out_port.open_port()
        self.m_image_out_port.add_attribute(f'{self.m_method}', similarities, static=False)
        self.m_image_out_port.close_port()


class SelectByAttributeModule(ProcessingModule):
    """
    Pipeline module for selecting frames based on attribute values.
    """

    __author__ = 'Benedikt Schmidhuber, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 selected_out_tag: str,
                 removed_out_tag: str,
                 attribute_tag: str,
                 number_frames: int = 100,
                 order: str = 'descending') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_tag : str
            Tag of the database entry that is read as input.
        selected_out_tag : str
            Tag of the database entry to which the selected frames are written.
        removed_out_tag : str
            Tag of the database entry to which the removed frames are written.
        attribute_tag : str
            Name of the attribute which is used to sort and select the frames.
        number_frames : int
            Number of frames that are selected.
        order : str
            Order in which the frames are selected. Can be either 'descending' (will select the
            lowest attribute values) or 'ascending' (will select the highest attribute values).

        Returns
        -------
        NoneType
            None

        Examples
        --------
        The example below selects the first 100 frames with an ascending order of the ``INDEX``
        values that are stored to the 'im_arr' dataset::

            SelectByAttributeModule(name_in='frame_selection',
                                    image_in_tag='im_arr',
                                    attribute_tag='INDEX',
                                    selected_frames=100,
                                    order='ascending',
                                    selected_out_tag='im_arr_selected',
                                    removed_out_tag='im_arr_removed'))

        The example below selects the 200 frames with the largest ``SSIM`` values that
        are stored to the 'im_arr' dataset::

            SelectByAttributeModule(name_in='frame_selection',
                                    image_in_tag='im_arr',
                                    attribute_tag='SSIM',
                                    selected_frames=200,
                                    order='descending',
                                    selected_out_tag='im_arr_selected',
                                    removed_out_tag='im_arr_removed'))
        """

        super(SelectByAttributeModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_selected_out_port = self.add_output_port(selected_out_tag)
        self.m_removed_out_port = self.add_output_port(removed_out_tag)

        if order not in ('ascending', 'descending'):
            raise ValueError('The selected order is not available. The available options are '
                             '\'ascending\' or \'descending\'.')

        self.m_attribute_tag = attribute_tag
        self.m_number_frames = number_frames
        self.m_order = order

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Selects images according to a specified attribute tag and
        ordering, e.g. the highest 150 ``INDEX`` frames, or the lowest 50 ``PCC`` frames.
        The order of the selected images is determined by the `descending` or `ascending`
        attribute values. To sort the images again by their original order, the
        :class:`~pynpoint.processing.psfpreparation.SortParangModule` can be used.

        Returns
        -------
        NoneType
            None
        """

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.del_all_data()
            self.m_selected_out_port.del_all_attributes()

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.del_all_data()
            self.m_removed_out_port.del_all_attributes()

        nimages = self.m_image_in_port.get_shape()[0]
        attribute = self.m_image_in_port.get_attribute(f'{self.m_attribute_tag}')

        if nimages != attribute.size:
            raise ValueError(f'The attribute {{self.m_attribute_tag}} does not have the same '
                             f'length ({len(attribute)}) as the tag has images ({nimages}). '
                             f'Please check the attribute you have chosen for selection.')

        if self.m_order == 'descending':
            # sort attribute in descending order
            sorting_order = np.argsort(attribute)[::-1]
        else:
            # sort attribute in ascending order
            sorting_order = np.argsort(attribute)

        index_del = sorting_order[self.m_number_frames:]

        write_selected_data(memory=self._m_config_port.get_attribute('MEMORY'),
                            indices=index_del,
                            image_in_port=self.m_image_in_port,
                            selected_out_port=self.m_selected_out_port,
                            removed_out_port=self.m_removed_out_port)

        write_selected_attributes(indices=index_del,
                                  image_in_port=self.m_image_in_port,
                                  selected_out_port=self.m_selected_out_port,
                                  removed_out_port=self.m_removed_out_port,
                                  module_type='SelectByAttributeModule',
                                  history=f'selected tag = {self.m_attribute_tag}')

        self.m_image_in_port.close_port()


class ResidualSelectionModule(ProcessingModule):
    """
    Pipeline module for applying a frame selection on the residuals of the PSF subtraction.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 selected_out_tag: str,
                 removed_out_tag: str,
                 percentage: float = 10.,
                 annulus_radii: Tuple[float, float] = (0.1, 0.2)) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        selected_out_tag : str
            Tag of the database entry with the selected images that are written as output.
        removed_out_tag : str
            Tag of the database entry with the removed images that are written as output.
        percentage : float
            The percentage of best frames that is selected.
        annulus_radii : tuple(float, float)
            Inner and outer radius of the annulus (arcsec).

        Returns
        -------
        NoneType
            None
        """

        super(ResidualSelectionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        self.m_selected_out_port = self.add_output_port(selected_out_tag)
        self.m_removed_out_port = self.add_output_port(removed_out_tag)

        self.m_percentage = percentage
        self.m_annulus_radii = annulus_radii

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Applies a frame selection on the derotated residuals from the
        PSF subtraction. The pixels within an annulus (e.g. at the separation of an expected
        planet) are selected and the standard deviation is calculated. The chosen percentage
        of images with the lowest standard deviation are stored as output.

        Returns
        -------
        NoneType
            None
        """

        self.m_selected_out_port.del_all_data()
        self.m_selected_out_port.del_all_attributes()

        self.m_removed_out_port.del_all_data()
        self.m_removed_out_port.del_all_attributes()

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        nimages = self.m_image_in_port.get_shape()[0]
        npix = self.m_image_in_port.get_shape()[-1]

        rr_grid = pixel_distance((npix, npix), position=None)

        pixel_select = np.where((rr_grid > self.m_annulus_radii[0]/pixscale) &
                                (rr_grid < self.m_annulus_radii[1]/pixscale))

        start_time = time.time()
        phot_annulus = np.zeros(nimages)

        for i in range(nimages):
            progress(i, nimages, 'Aperture photometry...', start_time)

            phot_annulus[i] = np.sum(np.abs(self.m_image_in_port[i][pixel_select]))

        print(f'Minimum, maximum = {np.amin(phot_annulus):.2f}, {np.amax(phot_annulus):.2f}')
        print(f'Mean, median = {np.nanmean(phot_annulus):.2f}, {np.nanmedian(phot_annulus):.2f}')
        print(f'Standard deviation = {np.nanstd(phot_annulus):.2f}')

        n_select = int(nimages*self.m_percentage/100.)
        index_del = np.argsort(phot_annulus)[n_select:]

        write_selected_data(memory=self._m_config_port.get_attribute('MEMORY'),
                            indices=index_del,
                            image_in_port=self.m_image_in_port,
                            selected_out_port=self.m_selected_out_port,
                            removed_out_port=self.m_removed_out_port)

        write_selected_attributes(indices=index_del,
                                  image_in_port=self.m_image_in_port,
                                  selected_out_port=self.m_selected_out_port,
                                  removed_out_port=self.m_removed_out_port,
                                  module_type='ResidualSelectionModule',
                                  history=f'frames removed = {index_del.size}')

        self.m_image_in_port.close_port()
