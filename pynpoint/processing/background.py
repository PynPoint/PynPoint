"""
Pipeline modules for subtraction of the background emission.
"""

import time
import warnings

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import create_mask
from pynpoint.util.module import progress


class SimpleBackgroundSubtractionModule(ProcessingModule):
    """
    Pipeline module for simple background subtraction. Only applicable on data obtained with
    dithering.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 shift: int) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        shift : int
            Frame index offset for the background subtraction. Typically equal to the number of
            frames per dither location.

        Returns
        -------
        NoneType
            None
        """

        super(SimpleBackgroundSubtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_shift = shift

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Simple background subtraction with a constant index offset.

        Returns
        -------
        NoneType
            None
        """

        nframes = self.m_image_in_port.get_shape()[0]

        subtract = self.m_image_in_port[0] - self.m_image_in_port[(0 + self.m_shift) % nframes]

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            self.m_image_out_port[0] = subtract
        else:
            self.m_image_out_port.set_all(subtract, data_dim=3)

        start_time = time.time()

        for i in range(1, nframes):
            progress(i, nframes, 'Subtracting background...', start_time)

            subtract = self.m_image_in_port[i] - self.m_image_in_port[(i + self.m_shift) % nframes]

            if self.m_image_in_port.tag == self.m_image_out_port.tag:
                self.m_image_out_port[i] = subtract
            else:
                self.m_image_out_port.append(subtract)

        history = f'shift = {self.m_shift}'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('SimpleBackgroundSubtractionModule', history)
        self.m_image_out_port.close_port()


class MeanBackgroundSubtractionModule(ProcessingModule):
    """
    Pipeline module for mean background subtraction. Only applicable on data obtained with
    dithering.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 shift: int = None,
                 cubes: int = 1) -> None:
        """
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.
        shift : int, None
            Image index offset for the background subtraction. Typically equal to the number of
            frames per dither location. If set to None, the ``NFRAMES`` attribute will be used to
            select the background frames automatically. The *cubes* parameters should be set when
            *shift* is set to None.
        cubes : int
            Number of consecutive cubes per dithering position.

        Returns
        -------
        NoneType
            None
        """

        super(MeanBackgroundSubtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_shift = shift
        self.m_cubes = cubes

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Mean background subtraction which uses either a constant index
        offset or the ``NFRAMES`` attributes. The mean background is calculated from the cubes
        before and after the science cube.

        Returns
        -------
        NoneType
            None
        """

        # Use NFRAMES values if shift=None
        if self.m_shift is None:
            self.m_shift = self.m_image_in_port.get_attribute('NFRAMES')

        nframes = self.m_image_in_port.get_shape()[0]

        if not isinstance(self.m_shift, np.ndarray) and nframes < self.m_shift*2.0:
            raise ValueError('The input stack is too small for a mean background subtraction. The '
                             'position of the star should shift at least once.')

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError('The tag of the input port should be different from the output port.')

        # Number of substacks
        if isinstance(self.m_shift, np.ndarray):
            nstacks = np.size(self.m_shift)
        else:
            nstacks = int(np.floor(nframes/self.m_shift))

        # First mean subtraction to set up the output port array
        if isinstance(self.m_shift, np.ndarray):
            next_start = np.sum(self.m_shift[0:self.m_cubes])
            next_end = np.sum(self.m_shift[0:2*self.m_cubes])

            if 2*self.m_cubes > np.size(self.m_shift):
                raise ValueError('Not enough frames available for the background subtraction.')

            bg_data = self.m_image_in_port[next_start:next_end, ]
            bg_mean = np.mean(bg_data, axis=0)

        else:
            bg_data = self.m_image_in_port[self.m_shift:2*self.m_shift, ]
            bg_mean = np.mean(bg_data, axis=0)

        # Initiate the result port data with the first frame
        bg_sub = self.m_image_in_port[0, ] - bg_mean
        self.m_image_out_port.set_all(bg_sub, data_dim=3)

        # Mean subtraction of the first stack (minus the first frame)
        if isinstance(self.m_shift, np.ndarray):
            tmp_data = self.m_image_in_port[1:next_start, ]
            tmp_data = tmp_data - bg_mean
            self.m_image_out_port.append(tmp_data)

        else:
            tmp_data = self.m_image_in_port[1:self.m_shift, ]
            tmp_data = tmp_data - bg_mean
            self.m_image_out_port.append(tmp_data)

        # Processing of the rest of the data
        start_time = time.time()
        if isinstance(self.m_shift, np.ndarray):
            for i in range(self.m_cubes, nstacks, self.m_cubes):
                progress(i, nstacks, 'Subtracting background...', start_time)

                prev_start = np.sum(self.m_shift[0:i-self.m_cubes])
                prev_end = np.sum(self.m_shift[0:i])

                next_start = np.sum(self.m_shift[0:i+self.m_cubes])
                next_end = np.sum(self.m_shift[0:i+2*self.m_cubes])

                # calc the mean (previous)
                tmp_data = self.m_image_in_port[prev_start:prev_end, ]
                tmp_mean = np.mean(tmp_data, axis=0)

                if i < nstacks-self.m_cubes:
                    # calc the mean (next)
                    tmp_data = self.m_image_in_port[next_start:next_end, ]
                    tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

                # subtract mean
                tmp_data = self.m_image_in_port[prev_end:next_start, ]
                tmp_data = tmp_data - tmp_mean
                self.m_image_out_port.append(tmp_data)

        else:
            # the last and the one before will be performed afterwards
            top = int(np.ceil(nframes/self.m_shift)) - 2

            for i in range(1, top, 1):
                progress(i, top, 'Subtracting background...', start_time)

                # calc the mean (next)
                tmp_data = self.m_image_in_port[(i+1)*self.m_shift:(i+2)*self.m_shift, ]
                tmp_mean = np.mean(tmp_data, axis=0)

                # calc the mean (previous)
                tmp_data = self.m_image_in_port[(i-1)*self.m_shift:(i+0)*self.m_shift, ]
                tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

                # subtract mean
                tmp_data = self.m_image_in_port[(i+0)*self.m_shift:(i+1)*self.m_shift, ]
                tmp_data = tmp_data - tmp_mean
                self.m_image_out_port.append(tmp_data)

            # last and the one before
            # 1. ------------------------------- one before -------------------
            # calc the mean (previous)
            tmp_data = self.m_image_in_port[(top-1)*self.m_shift:(top+0)*self.m_shift, ]
            tmp_mean = np.mean(tmp_data, axis=0)

            # calc the mean (next)
            # 'nframes' is important if the last step is to huge
            tmp_data = self.m_image_in_port[(top+1)*self.m_shift:nframes, ]
            tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

            # subtract mean
            tmp_data = self.m_image_in_port[top*self.m_shift:(top+1)*self.m_shift, ]
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)

            # 2. ------------------------------- last -------------------
            # calc the mean (previous)
            tmp_data = self.m_image_in_port[(top+0)*self.m_shift:(top+1)*self.m_shift, ]
            tmp_mean = np.mean(tmp_data, axis=0)

            # subtract mean
            tmp_data = self.m_image_in_port[(top+1)*self.m_shift:nframes, ]
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)
            # -----------------------------------------------------------

        if isinstance(self.m_shift, np.ndarray):
            history = f'shift = NFRAMES'
        else:
            history = f'shift = {self.m_shift}'

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('MeanBackgroundSubtractionModule', history)
        self.m_image_out_port.close_port()


class LineSubtractionModule(ProcessingModule):
    """
    Pipeline module for subtracting the background emission from each pixel by computing the mean
    or median of all values in the row and column of the pixel. The module can for example be
    used if no background data is available or to remove a detector bias.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 combine: str = 'median',
                 mask=None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        combine : str
            The method by which the column and row pixel values are combined ('median' or 'mean').
            Using a mean-combination is computationally faster than a median-combination.
        mask : float, None
            The radius of the mask within which pixel values are ignored. No mask is used if set
            to None.

        Returns
        -------
        NoneType
            None
        """

        super(LineSubtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_combine = combine
        self.m_mask = mask

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Selects the pixel values in the column and row at each pixel
        position, computes the mean or median value while excluding pixels within the radius of
        the mask, and subtracts the mean or median value from each pixel separately.

        Returns
        -------
        NoneType
            None
        """

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        im_shape = self.m_image_in_port.get_shape()[-2:]

        def _subtract_line(image_in, mask):
            image_tmp = np.copy(image_in)
            image_tmp[mask == 0.] = np.nan

            if self.m_combine == 'mean':
                row_mean = np.nanmean(image_tmp, axis=1)
                col_mean = np.nanmean(image_tmp, axis=0)

                x_grid, y_grid = np.meshgrid(col_mean, row_mean)
                subtract = (x_grid+y_grid)/2.

            elif self.m_combine == 'median':
                subtract = np.zeros(im_shape)

                col_median = np.nanmedian(image_tmp, axis=0)
                col_2d = np.tile(col_median, (im_shape[1], 1))

                image_tmp -= col_2d
                image_tmp[mask == 0.] = np.nan

                row_median = np.nanmedian(image_tmp, axis=1)
                row_2d = np.tile(row_median, (im_shape[0], 1))
                row_2d = np.rot90(row_2d)  # 90 deg rotation in clockwise direction

                subtract = col_2d + row_2d

            return image_in - subtract

        if self.m_mask:
            size = (self.m_mask/pixscale, None)
        else:
            size = (None, None)

        mask = create_mask(im_shape, size)

        self.apply_function_to_images(_subtract_line,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Background subtraction',
                                      func_args=(mask, ))

        history = f'combine = {self.m_combine}'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('LineSubtractionModule', history)
        self.m_image_out_port.close_port()


class NoddingBackgroundModule(ProcessingModule):
    """
    Pipeline module for background subtraction of data obtained with nodding (e.g., NACO AGPM
    data). Before using this module, the sky images should be stacked with the StackCubesModule
    such that each image in the stack of sky images corresponds to the mean combination of a
    single FITS data cube.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 science_in_tag: str,
                 sky_in_tag: str,
                 image_out_tag: str,
                 mode: str = 'both') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        science_in_tag : str
            Tag of the database entry with science images that are read as input.
        sky_in_tag : str
            Tag of the database entry with sky images that are read as input. The
            :class:`~pynpoint.processing.stacksubset.StackCubesModule` should be used on the sky
            images beforehand.
        image_out_tag : str
            Tag of the database entry with sky subtracted images that are written as output.
        mode : str
            Sky images that are subtracted, relative to the science images. Either the next,
            previous, or average of the next and previous cubes of sky frames can be used by
            choosing 'next', 'previous', or 'both', respectively.

        Returns
        -------
        NoneType
            None
        """

        super(NoddingBackgroundModule, self).__init__(name_in=name_in)

        self.m_science_in_port = self.add_input_port(science_in_tag)
        self.m_sky_in_port = self.add_input_port(sky_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_time_stamps = []

        if mode in ['next', 'previous', 'both']:
            self.m_mode = mode
        else:
            raise ValueError('Mode needs to be \'next\', \'previous\', or \'both\'.')

    def _create_time_stamp_list(self):
        """
        Internal method for assigning a time stamp, based on the exposure number ID, to each cube
        of sky and science images.
        """

        class TimeStamp:
            """
            Class for creating a time stamp.
            """

            def __init__(self, time_in, im_type, index):

                self.m_time = time_in
                self.m_im_type = im_type
                self.m_index = index

            def __repr__(self):

                return repr((self.m_time,
                             self.m_im_type,
                             self.m_index))

        exp_no_sky = self.m_sky_in_port.get_attribute('EXP_NO')
        exp_no_science = self.m_science_in_port.get_attribute('EXP_NO')

        nframes_sky = self.m_sky_in_port.get_attribute('NFRAMES')
        nframes_science = self.m_science_in_port.get_attribute('NFRAMES')

        if np.all(nframes_sky != 1):
            warnings.warn('The NFRAMES values of the sky images are not all equal to unity. '
                          'The StackCubesModule should be applied on the sky images before the '
                          'NoddingBackgroundModule is used.')

        for i, item in enumerate(exp_no_sky):
            self.m_time_stamps.append(TimeStamp(item, 'SKY', i))

        current = 0
        for i, item in enumerate(exp_no_science):
            frames = slice(current, current+nframes_science[i])
            self.m_time_stamps.append(TimeStamp(item, 'SCIENCE', frames))
            current += nframes_science[i]

        self.m_time_stamps = sorted(self.m_time_stamps, key=lambda time_stamp: time_stamp.m_time)

    def calc_sky_frame(self,
                       index_of_science_data):
        """
        Method for finding the required sky frame (next, previous, or the mean of next and
        previous) by comparing the time stamp of the science frame with preceding and following
        sky frames.
        """

        if not any(x.m_im_type == 'SKY' for x in self.m_time_stamps):
            raise ValueError('List of time stamps does not contain any SKY images.')

        def search_for_next_sky():
            for i in range(index_of_science_data, len(self.m_time_stamps)):
                if self.m_time_stamps[i].m_im_type == 'SKY':
                    return self.m_sky_in_port[self.m_time_stamps[i].m_index, ]

            # no next sky found, look for previous sky
            return search_for_previous_sky()

        def search_for_previous_sky():
            for i in reversed(list(range(0, index_of_science_data))):
                if self.m_time_stamps[i].m_im_type == 'SKY':
                    return self.m_sky_in_port[self.m_time_stamps[i].m_index, ]

            # no previous sky found, look for next sky
            return search_for_next_sky()

        if self.m_mode == 'next':
            return search_for_next_sky()

        if self.m_mode == 'previous':
            return search_for_previous_sky()

        if self.m_mode == 'both':
            previous_sky = search_for_previous_sky()
            next_sky = search_for_next_sky()

            return (previous_sky+next_sky)/2.

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Create list of time stamps, get sky and science images, and
        subtract the sky images from the science images.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        self._create_time_stamp_list()

        start_time = time.time()
        for i, time_entry in enumerate(self.m_time_stamps):
            progress(i, len(self.m_time_stamps), 'Subtracting background...', start_time)

            if time_entry.m_im_type == 'SKY':
                continue

            sky = self.calc_sky_frame(i)
            science = self.m_science_in_port[time_entry.m_index, ]

            self.m_image_out_port.append(science - sky[None, ], data_dim=3)

        history = f'mode = {self.m_mode}'
        self.m_image_out_port.copy_attributes(self.m_science_in_port)
        self.m_image_out_port.add_history('NoddingBackgroundModule', history)
        self.m_image_out_port.close_port()
