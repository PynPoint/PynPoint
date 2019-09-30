"""
Pipeline modules to prepare the data for the PSF subtraction.
"""

import time
import warnings

from typing import Tuple

import numpy as np

from astropy.coordinates import EarthLocation
from astropy.time import Time
from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress, memory_frames
from pynpoint.util.image import create_mask, scale_image, shift_image


class PSFpreparationModule(ProcessingModule):
    """
    Module to prepare the data for PSF subtraction with PCA. The preparation steps include masking
    and image normalization.
    """

    __author__ = 'Markus Bonse, Tomas Stolker, Timothy Gebhard'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 mask_out_tag: str = None,
                 norm: bool = False,
                 resize: float = None,
                 cent_size: float = None,
                 edge_size: float = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry with images that is written as output.
        mask_out_tag : str, None, optional
            Tag of the database entry with the mask that is written as output. If set to None, no
            mask array is saved.
        norm : bool
            Normalize each image by its Frobenius norm.
        resize : float, None
            DEPRECATED. This parameter is currently ignored by the module and will be removed in a
            future version of PynPoint.
        cent_size : float, None, optional
            Radius of the central mask (in arcsec). No mask is used when set to None.
        edge_size : float, None, optional
            Outer radius (in arcsec) beyond which pixels are masked. No outer mask is used when set
            to None. If the value is larger than half the image size then it will be set to half
            the image size.

        Returns
        -------
        NoneType
            None
        """

        super(PSFpreparationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if mask_out_tag is None:
            self.m_mask_out_port = None
        else:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_norm = norm

        # Raise a DeprecationWarning if the resize argument is used
        if resize is not None:
            warnings.warn('The \'resize\' parameter has been deprecated. Its value is currently '
                          'being ignored, and the argument will be removed in a future version '
                          'of PynPoint.', DeprecationWarning)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Masks and normalizes the images.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_mask_out_port is not None:
            self.m_mask_out_port.del_all_data()
            self.m_mask_out_port.del_all_attributes()

        # Get PIXSCALE and MEMORY attributes
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        memory = self._m_config_port.get_attribute('MEMORY')

        # Get the number of images and split into batches to comply with memory constraints
        im_shape = self.m_image_in_port.get_shape()
        nimages = im_shape[0]
        frames = memory_frames(memory, nimages)

        # Convert m_cent_size and m_edge_size from arcseconds to pixels
        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale
        if self.m_edge_size is not None:
            self.m_edge_size /= pixscale

        # Create 2D disk mask which will be applied to every frame
        mask = create_mask((int(im_shape[-2]), int(im_shape[-1])),
                           (self.m_cent_size, self.m_edge_size)).astype(bool)

        # Keep track of the normalization vectors in case we are normalizing the images (if
        # we are not normalizing, this list will remain empty)
        norms = list()

        # Run the PSFpreparationModule for each subset of frames
        start_time = time.time()
        for i, _ in enumerate(frames[:-1]):

            # Print progress to command line
            progress(i, len(frames[:-1]), 'Preparing images for PSF subtraction...', start_time)

            # Get the images and ensure they have the correct 3D shape with the following
            # three dimensions: (batch_size, height, width)
            images = self.m_image_in_port[frames[i]:frames[i+1], ]

            if images.ndim == 2:
                warnings.warn('The input data has 2 dimensions whereas 3 dimensions are required. '
                              'An extra dimension has been added.')

                images = images[np.newaxis, ...]

            # Apply the mask, i.e., set all pixels to 0 where the mask is False
            images[:, ~mask] = 0.

            # If desired, normalize the images using the Frobenius norm
            if self.m_norm:
                im_norm = np.linalg.norm(images, ord='fro', axis=(1, 2))
                images /= im_norm[:, np.newaxis, np.newaxis]
                norms.append(im_norm)

            # Write processed images to output port
            self.m_image_out_port.append(images, data_dim=3)

        # Store information about mask
        if self.m_mask_out_port is not None:
            self.m_mask_out_port.set_all(mask)
            self.m_mask_out_port.copy_attributes(self.m_image_in_port)

        # Copy attributes from input port
        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        # If the norms list is not empty (i.e., if we have computed the norm for every image),
        # we can also save the corresponding norm vector as an additional attribute
        if norms:
            self.m_image_out_port.add_attribute(name='norm',
                                                value=np.hstack(norms),
                                                static=False)

        # Save cent_size and edge_size as attributes to the output port
        if self.m_cent_size is not None:
            self.m_image_out_port.add_attribute(name='cent_size',
                                                value=self.m_cent_size * pixscale,
                                                static=True)
        if self.m_edge_size is not None:
            self.m_image_out_port.add_attribute(name='edge_size',
                                                value=self.m_edge_size * pixscale,
                                                static=True)


class AngleInterpolationModule(ProcessingModule):
    """
    Module for calculating the parallactic angle values by interpolating between the begin and end
    value of a data cube.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 data_tag: str) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        data_tag : str
            Tag of the database entry for which the parallactic angles are written as attributes.

        Returns
        -------
        NoneType
            None
        """

        super(AngleInterpolationModule, self).__init__(name_in)

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Calculates the parallactic angles of each frame by linearly
        interpolating between the start and end values of the data cubes. The values are written
        as attributes to *data_tag*. A correction of 360 deg is applied when the start and end
        values of the angles change sign at +/-180 deg.

        Returns
        -------
        NoneType
            None
        """

        parang_start = self.m_data_in_port.get_attribute('PARANG_START')
        parang_end = self.m_data_in_port.get_attribute('PARANG_END')

        steps = self.m_data_in_port.get_attribute('NFRAMES')

        if 'NDIT' in self.m_data_in_port.get_all_non_static_attributes():
            ndit = self.m_data_in_port.get_attribute('NDIT')

            if not np.all(ndit == steps):
                warnings.warn('There is a mismatch between the NDIT and NFRAMES values. The '
                              'parallactic angles are calculated with a linear interpolation by '
                              'using NFRAMES steps. A frame selection should be applied after '
                              'the parallactic angles are calculated.')

        new_angles = []

        start_time = time.time()
        for i, _ in enumerate(parang_start):
            progress(i, len(parang_start), 'Interpolating parallactic angles...', start_time)

            if parang_start[i] < -170. and parang_end[i] > 170.:
                parang_start[i] += 360.

            elif parang_end[i] < -170. and parang_start[i] > 170.:
                parang_end[i] += 360.

            if steps[i] == 1:
                new_angles = np.append(new_angles,
                                       [(parang_start[i] + parang_end[i])/2.])

            elif steps[i] != 1:
                new_angles = np.append(new_angles,
                                       np.linspace(parang_start[i],
                                                   parang_end[i],
                                                   num=steps[i]))

        self.m_data_out_port.add_attribute('PARANG',
                                           new_angles,
                                           static=False)


class SortParangModule(ProcessingModule):
    """
    Module to sort the images and non-static attributes with increasing INDEX.
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
            Tag of the database entry with images that is written as output. Should be different
            from *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(SortParangModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Sorts the images and relevant non-static attributes.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError('Input and output port should have a different tag.')

        memory = self._m_config_port.get_attribute('MEMORY')
        index = self.m_image_in_port.get_attribute('INDEX')

        index_new = np.zeros(index.shape, dtype=np.int)

        if 'PARANG' in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute('PARANG')
            parang_new = np.zeros(parang.shape)

        else:
            parang_new = None

        if 'STAR_POSITION' in self.m_image_in_port.get_all_non_static_attributes():
            star = self.m_image_in_port.get_attribute('STAR_POSITION')
            star_new = np.zeros(star.shape)

        else:
            star_new = None

        index_sort = np.argsort(index)

        nimages = self.m_image_in_port.get_shape()[0]

        frames = memory_frames(memory, nimages)

        start_time = time.time()
        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Sorting images in time...', start_time)

            index_new[frames[i]:frames[i+1]] = index[index_sort[frames[i]:frames[i+1]]]

            if parang_new is not None:
                parang_new[frames[i]:frames[i+1]] = parang[index_sort[frames[i]:frames[i+1]]]

            if star_new is not None:
                star_new[frames[i]:frames[i+1]] = star[index_sort[frames[i]:frames[i+1]]]

            # h5py indexing elements must be in increasing order
            for _, item in enumerate(index_sort[frames[i]:frames[i+1]]):
                self.m_image_out_port.append(self.m_image_in_port[item, ], data_dim=3)

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('SortParangModule', 'sorted by INDEX')
        self.m_image_out_port.add_attribute('INDEX', index_new, static=False)

        if parang_new is not None:
            self.m_image_out_port.add_attribute('PARANG', parang_new, static=False)

        if star_new is not None:
            self.m_image_out_port.add_attribute('STAR_POSITION', star_new, static=False)

        self.m_image_out_port.close_port()


class AngleCalculationModule(ProcessingModule):
    """
    Module for calculating the parallactic angles. The start time of the observation is taken and
    multiples of the exposure time are added to derive the parallactic angle of each frame inside
    the cube. Instrument specific overheads are included.
    """

    __author__ = 'Alexander Bohn, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 data_tag: str,
                 instrument: str = 'NACO') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        data_tag : str
            Tag of the database entry for which the parallactic angles are written as attributes.
        instrument : str
            Instrument name ('NACO', 'SPHERE/IRDIS', or 'SPHERE/IFS').

        Returns
        -------
        NoneType
            None
        """

        super(AngleCalculationModule, self).__init__(name_in)

        # Parameters
        self.m_instrument = instrument

        # Set parameters according to choice of instrument
        if self.m_instrument == 'NACO':

            # pupil offset in degrees
            self.m_pupil_offset = 0.            # No offset here

            # no overheads in cube mode, since cube is read out after all individual exposures
            # see NACO manual page 62 (v102)
            self.m_O_START = 0.
            self.m_DIT_DELAY = 0.
            self.m_ROT = 0.

            # rotator offset in degrees
            self.m_rot_offset = 89.44           # According to NACO manual page 65 (v102)

        elif self.m_instrument == 'SPHERE/IRDIS':

            # pupil offset in degrees
            self.m_pupil_offset = -135.99       # According to SPHERE manual page 64 (v102)

            # overheads in cube mode (several NDITS) in hours
            self.m_O_START = 0.3 / 3600.        # According to SPHERE manual page 90/91 (v102)
            self.m_DIT_DELAY = 0.1 / 3600.      # According to SPHERE manual page 90/91 (v102)
            self.m_ROT = 0.838 / 3600.          # According to SPHERE manual page 90/91 (v102)

            # rotator offset in degrees
            self.m_rot_offset = 0.              # no offset here

        elif self.m_instrument == 'SPHERE/IFS':

            # pupil offset in degrees
            self.m_pupil_offset = -135.99 - 100.48  # According to SPHERE manual page 64 (v102)

            # overheads in cube mode (several NDITS) in hours
            self.m_O_START = 0.3 / 3600.            # According to SPHERE manual page 90/91 (v102)
            self.m_DIT_DELAY = 0.2 / 3600.          # According to SPHERE manual page 90/91 (v102)
            self.m_ROT = 1.65 / 3600.               # According to SPHERE manual page 90/91 (v102)

            # rotator offset in degrees
            self.m_rot_offset = 0.                  # no offset here

        else:
            raise ValueError('The instrument argument should be set to either \'NACO\', '
                             '\'SPHERE/IRDIS\', or \'SPHERE/IFS\'.')

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

    def _attribute_check(self, ndit, steps):

        if not np.all(ndit == steps):
            warnings.warn('There is a mismatch between the NDIT and NFRAMES values. A frame '
                          'selection should be applied after the parallactic angles are '
                          'calculated.')

        if self.m_instrument == 'SPHERE/IFS':
            warnings.warn('AngleCalculationModule has not been tested for SPHERE/IFS data.')

        if self.m_instrument in ('SPHERE/IRDIS', 'SPHERE/IFS'):

            if self._m_config_port.get_attribute('RA') != 'ESO INS4 DROT2 RA':

                warnings.warn('For SPHERE data it is recommended to use the header keyword '
                              '\'ESO INS4 DROT2 RA\' to specify the object\'s right ascension. '
                              'The input will be parsed accordingly. Using the regular '
                              '\'RA\' keyword will lead to wrong parallactic angles.')

            if self._m_config_port.get_attribute('DEC') != 'ESO INS4 DROT2 DEC':

                warnings.warn('For SPHERE data it is recommended to use the header keyword '
                              '\'ESO INS4 DROT2 DEC\' to specify the object\'s declination. '
                              'The input will be parsed accordingly. Using the regular '
                              '\'DEC\' keyword will lead to wrong parallactic angles.')

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Calculates the parallactic angles from the position of the object
        on the sky and the telescope location on earth. The start of the observation is used to
        extrapolate for the observation time of each individual image of a data cube. The values
        are written as PARANG attributes to *data_tag*.

        Returns
        -------
        NoneType
            None
        """

        # Load cube sizes
        steps = self.m_data_in_port.get_attribute('NFRAMES')
        ndit = self.m_data_in_port.get_attribute('NDIT')

        self._attribute_check(ndit, steps)

        # Load exposure time [hours]
        exptime = self.m_data_in_port.get_attribute('DIT')/3600.

        # Load telescope location
        tel_lat = self.m_data_in_port.get_attribute('LATITUDE')
        tel_lon = self.m_data_in_port.get_attribute('LONGITUDE')

        # Load temporary target position
        tmp_ra = self.m_data_in_port.get_attribute('RA')
        tmp_dec = self.m_data_in_port.get_attribute('DEC')

        # Parse to degree depending on instrument
        if 'SPHERE' in self.m_instrument:

            # get sign of declination
            tmp_dec_sign = np.sign(tmp_dec)
            tmp_dec = np.abs(tmp_dec)

            # parse RA
            tmp_ra_s = tmp_ra % 100
            tmp_ra_m = ((tmp_ra - tmp_ra_s) / 1e2) % 100
            tmp_ra_h = ((tmp_ra - tmp_ra_s - tmp_ra_m * 1e2) / 1e4)

            # parse DEC
            tmp_dec_s = tmp_dec % 100
            tmp_dec_m = ((tmp_dec - tmp_dec_s) / 1e2) % 100
            tmp_dec_d = ((tmp_dec - tmp_dec_s - tmp_dec_m * 1e2) / 1e4)

            # get RA and DEC in degree
            ra = (tmp_ra_h + tmp_ra_m / 60. + tmp_ra_s / 3600.) * 15.
            dec = tmp_dec_sign * (tmp_dec_d + tmp_dec_m / 60. + tmp_dec_s / 3600.)

        else:
            ra = tmp_ra
            dec = tmp_dec

        # Load start times of exposures
        obs_dates = self.m_data_in_port.get_attribute('DATE')

        # Load pupil positions during observations
        if self.m_instrument == 'NACO':
            pupil_pos = self.m_data_in_port.get_attribute('PUPIL')

        elif self.m_instrument == 'SPHERE/IRDIS':
            pupil_pos = np.zeros(steps.shape)

        elif self.m_instrument == 'SPHERE/IFS':
            pupil_pos = np.zeros(steps.shape)

        new_angles = np.array([])
        pupil_pos_arr = np.array([])

        start_time = time.time()

        # Calculate parallactic angles for each cube
        for i, tmp_steps in enumerate(steps):
            progress(i, len(steps), 'Calculating parallactic angles...', start_time)

            t = Time(obs_dates[i].decode('utf-8'),
                     location=EarthLocation(lat=tel_lat, lon=tel_lon))

            sid_time = t.sidereal_time('apparent').value

            # Extrapolate sideral times from start time of the cube for each frame of it
            sid_time_arr = np.linspace(sid_time+self.m_O_START,
                                       (sid_time+self.m_O_START) +
                                       (exptime+self.m_DIT_DELAY + self.m_ROT)*(tmp_steps-1),
                                       tmp_steps)

            # Convert to degrees
            sid_time_arr_deg = sid_time_arr * 15.

            # Calculate hour angle in degrees
            hour_angle = sid_time_arr_deg - ra[i]

            # Conversion to radians:
            hour_angle_rad = np.deg2rad(hour_angle)
            dec_rad = np.deg2rad(dec[i])
            lat_rad = np.deg2rad(tel_lat)

            p_angle = np.arctan2(np.sin(hour_angle_rad),
                                 (np.cos(dec_rad)*np.tan(lat_rad) -
                                  np.sin(dec_rad)*np.cos(hour_angle_rad)))

            new_angles = np.append(new_angles, np.rad2deg(p_angle))
            pupil_pos_arr = np.append(pupil_pos_arr, np.ones(tmp_steps)*pupil_pos[i])

        # Correct for rotator (SPHERE) or pupil offset (NACO)
        if self.m_instrument == 'NACO':
            # See NACO manual page 65 (v102)
            new_angles_corr = new_angles - (90. + (self.m_rot_offset-pupil_pos_arr))

        elif self.m_instrument == 'SPHERE/IRDIS':
            # See SPHERE manual page 64 (v102)
            new_angles_corr = new_angles - self.m_pupil_offset

        elif self.m_instrument == 'SPHERE/IFS':
            # See SPHERE manual page 64 (v102)
            new_angles_corr = new_angles - self.m_pupil_offset

        indices = np.where(new_angles_corr < -180.)[0]
        if indices.size > 0:
            new_angles_corr[indices] += 360.

        indices = np.where(new_angles_corr > 180.)[0]
        if indices.size > 0:
            new_angles_corr[indices] -= 360.

        self.m_data_out_port.add_attribute('PARANG', new_angles_corr, static=False)


class SDIpreparationModule(ProcessingModule):
    """
    Module for preparing continuum frames for SDI subtraction.
    """

    __author__ = 'Gabriele Cugno, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 wavelength: Tuple[float, float],
                 width: Tuple[float, float]) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.
        wavelength : tuple(float, float)
            The central wavelengths of the line and continuum filter, (line, continuum), in
            arbitrary but identical units.
        width : tuple(float, float)
            The equivalent widths of the line and continuum filter, (line, continuum), in
            arbitrary but identical units.

        Returns
        -------
        NoneType
            None
        """

        super(SDIpreparationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_line_wvl = wavelength[0]
        self.m_cnt_wvl = wavelength[1]

        self.m_line_width = width[0]
        self.m_cnt_width = width[1]

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Normalizes the images for the different filter widths,
        upscales the images, and crops the images to the initial image shape in order to
        align the PSF patterns.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        wvl_factor = self.m_line_wvl/self.m_cnt_wvl
        width_factor = self.m_line_width/self.m_cnt_width

        nimages = self.m_image_in_port.get_shape()[0]

        start_time = time.time()
        for i in range(nimages):
            progress(i, nimages, 'Preparing images for SDI...', start_time)

            image = self.m_image_in_port[i, ]

            im_scale = width_factor * scale_image(image, wvl_factor, wvl_factor)

            if i == 0:
                npix_del = im_scale.shape[-1] - image.shape[-1]

                if npix_del % 2 == 0:
                    npix_del_a = int(npix_del/2)
                    npix_del_b = int(npix_del/2)

                else:
                    npix_del_a = int((npix_del-1)/2)
                    npix_del_b = int((npix_del+1)/2)

            im_crop = im_scale[npix_del_a:-npix_del_b, npix_del_a:-npix_del_b]

            if npix_del % 2 == 1:
                im_crop = shift_image(im_crop, (-0.5, -0.5), interpolation='spline')

            self.m_image_out_port.append(im_crop, data_dim=3)

        history = f'(line, continuum) = ({self.m_line_wvl}, {self.m_cnt_wvl})'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('SDIpreparationModule', history)
        self.m_image_in_port.close_port()
