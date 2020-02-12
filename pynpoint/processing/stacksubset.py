"""
Pipeline modules for stacking and subsampling of images.
"""

import time
import warnings

from typing import List

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress, memory_frames, stack_angles, angle_average
from pynpoint.util.image import rotate_images


class StackAndSubsetModule(ProcessingModule):
    """
    Pipeline module for stacking subsets of images and/or selecting a random sample of images.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 random: int = None,
                 stacking: int = None,
                 combine: str = 'mean',
                 max_rotation: float = None) -> None:
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
        random : int, None
            Number of random images. All images are used if set to None.
        stacking : int, None
            Number of stacked images per subset. No stacking is applied if set to None.
        combine : str
            Method for combining images ('mean' or 'median'). The angles are always mean-combined.
        max_rotation : float, None
            Maximum allowed field rotation throughout each subset of stacked images when
            `stacking` is not None. No restriction on the field rotation is applied if set to
            None.

        Returns
        -------
        NoneType
            None
        """

        super(StackAndSubsetModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_random = random
        self.m_stacking = stacking
        self.m_combine = combine
        self.m_max_rotation = max_rotation

        if self.m_stacking is None and self.m_random is None:
            warnings.warn('Both \'stacking\' and \'random\' are set to None.')

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Stacks subsets of images and/or selects a random subset. Also
        the parallactic angles are mean-combined if images are stacked.

        Returns
        -------
        NoneType
            None
        """

        def _stack_subsets(nimages, im_shape, parang):
            im_new = None
            parang_new = None

            if self.m_stacking is not None:
                if self.m_max_rotation is not None:
                    frames = stack_angles(self.m_stacking, parang, self.m_max_rotation)
                else:
                    frames = memory_frames(self.m_stacking, nimages)

                nimages_new = np.size(frames)-1

                if parang is None:
                    parang_new = None
                else:
                    parang_new = np.zeros(nimages_new)

                im_new = np.zeros((nimages_new, im_shape[1], im_shape[2]))

                start_time = time.time()

                for i in range(nimages_new):
                    progress(i, nimages_new, 'Stacking subsets of images...', start_time)

                    if parang is not None:
                        # parang_new[i] = np.mean(parang[frames[i]:frames[i+1]])
                        parang_new[i] = angle_average(parang[frames[i]:frames[i+1]])

                    im_subset = self.m_image_in_port[frames[i]:frames[i+1], ]

                    if self.m_combine == 'mean':
                        im_new[i, ] = np.mean(im_subset, axis=0)
                    elif self.m_combine == 'median':
                        im_new[i, ] = np.median(im_subset, axis=0)

                im_shape = im_new.shape

            else:
                if parang is not None:
                    parang_new = np.copy(parang)

            return im_shape, im_new, parang_new

        def _random_subset(im_shape, im_new, parang_new):
            if self.m_random is not None:
                choice = np.random.choice(im_shape[0], self.m_random, replace=False)
                choice = list(np.sort(choice))

                if parang_new is None:
                    parang_new = None
                else:
                    parang_new = parang_new[choice]

                if self.m_stacking is None:
                    im_new = self.m_image_in_port[list(choice), ]
                else:
                    im_new = im_new[choice, ]

            if self.m_random is None and self.m_stacking is None:
                nimages = 0
            elif im_new.ndim == 2:
                nimages = 1
            elif im_new.ndim == 3:
                nimages = im_new.shape[0]

            return nimages, im_new, parang_new

        non_static = self.m_image_in_port.get_all_non_static_attributes()

        im_shape = self.m_image_in_port.get_shape()
        nimages = im_shape[0]

        if self.m_random is not None:
            if self.m_stacking is None and im_shape[0] < self.m_random:
                raise ValueError('The number of images of the destination subset is larger than '
                                 'the number of images in the source.')

            if self.m_stacking is not None and \
                    int(float(im_shape[0])/float(self.m_stacking)) < self.m_random:
                raise ValueError('The number of images of the destination subset is larger than '
                                 'the number of images in the stacked source.')

        if 'PARANG' in non_static:
            parang = self.m_image_in_port.get_attribute('PARANG')
        else:
            parang = None

        im_shape, im_new, parang_new = _stack_subsets(nimages, im_shape, parang)
        nimages, im_new, parang_new = _random_subset(im_shape, im_new, parang_new)

        if self.m_random or self.m_stacking:
            self.m_image_out_port.set_all(im_new, keep_attributes=True)
            self.m_image_out_port.copy_attributes(self.m_image_in_port)
            self.m_image_out_port.add_attribute('INDEX', np.arange(0, nimages, 1), static=False)

            if parang_new is not None:
                self.m_image_out_port.add_attribute('PARANG', parang_new, static=False)

            if 'NFRAMES' in non_static:
                self.m_image_out_port.del_attribute('NFRAMES')

            history = f'stacking = {self.m_stacking}, random = {self.m_random}'
            self.m_image_out_port.add_history('StackAndSubsetModule', history)

        self.m_image_out_port.close_port()


class StackCubesModule(ProcessingModule):
    """
    Pipeline module for calculating the mean or median of each original data cube associated with a
    database tag.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 combine: str = 'mean') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry with the mean or median collapsed images that are written
            as output. Should be different from *image_in_tag*.
        combine : str
            Method to combine the images ('mean' or 'median').

        Returns
        -------
        NoneType
            None
        """

        super(StackCubesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_combine = combine

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Uses the NFRAMES attribute to select the images of each cube,
        calculates the mean or median of each cube, and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError('Input and output port should have a different tag.')

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        non_static = self.m_image_in_port.get_all_non_static_attributes()
        nframes = self.m_image_in_port.get_attribute('NFRAMES')

        if 'PARANG' in non_static:
            parang = self.m_image_in_port.get_attribute('PARANG')
        else:
            parang = None

        current = 0
        parang_new = []

        start_time = time.time()
        for i, frames in enumerate(nframes):
            progress(i, len(nframes), 'Stacking images per FITS cube...', start_time)

            if self.m_combine == 'mean':
                im_stack = np.mean(self.m_image_in_port[current:current+frames, ], axis=0)
            elif self.m_combine == 'median':
                im_stack = np.median(self.m_image_in_port[current:current+frames, ], axis=0)

            self.m_image_out_port.append(im_stack, data_dim=3)

            if parang is not None:
                parang_new.append(np.mean(parang[current:current+frames]))

            current += frames

        nimages = np.size(nframes)

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        if 'INDEX' in non_static:
            index = np.arange(0, nimages, 1, dtype=np.int)
            self.m_image_out_port.add_attribute('INDEX', index, static=False)

        if 'NFRAMES' in non_static:
            nframes = np.ones(nimages, dtype=np.int)
            self.m_image_out_port.add_attribute('NFRAMES', nframes, static=False)

        if 'PARANG' in non_static:
            self.m_image_out_port.add_attribute('PARANG', parang_new, static=False)

        self.m_image_out_port.close_port()


class DerotateAndStackModule(ProcessingModule):
    """
    Pipeline module for derotating and/or stacking (i.e., taking the median or average) of the
    images.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 derotate: bool = True,
                 stack: str = None,
                 extra_rot: float = 0.) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. The output is either 2D
            (*stack=False*) or 3D (*stack=True*).
        derotate : bool
            Derotate the images with the PARANG attribute.
        stack : str
            Type of stacking applied after optional derotation ('mean', 'median', or None for no
            stacking).
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).

        Returns
        -------
        NoneType
            None
        """

        super(DerotateAndStackModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_derotate = derotate
        self.m_stack = stack
        self.m_extra_rot = extra_rot

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Uses the PARANG attributes to derotate the images (if *derotate*
        is set to True) and applies an optional mean or median stacking afterwards.

        Returns
        -------
        NoneType
            None
        """

        def _initialize(ndim, npix):
            if ndim == 2:
                nimages = 1
            elif ndim == 3:
                nimages = self.m_image_in_port.get_shape()[0]

            if self.m_stack == 'median':
                frames = [0, nimages]
            else:
                frames = memory_frames(memory, nimages)

            if self.m_stack == 'mean':
                im_tot = np.zeros((npix, npix))
            else:
                im_tot = None

            return nimages, frames, im_tot

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError('Input and output port should have a different tag.')

        memory = self._m_config_port.get_attribute('MEMORY')

        if self.m_derotate:
            parang = self.m_image_in_port.get_attribute('PARANG')

        ndim = self.m_image_in_port.get_ndim()
        npix = self.m_image_in_port.get_shape()[1]

        nimages, frames, im_tot = _initialize(ndim, npix)

        start_time = time.time()
        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Derotating and/or stacking images...', start_time)

            images = self.m_image_in_port[frames[i]:frames[i+1], ]

            if self.m_derotate:
                angles = -parang[frames[i]:frames[i+1]]+self.m_extra_rot
                images = rotate_images(images, angles)

            if self.m_stack is None:
                if ndim == 2:
                    self.m_image_out_port.set_all(images[np.newaxis, ...])
                elif ndim == 3:
                    self.m_image_out_port.append(images, data_dim=3)

            elif self.m_stack == 'mean':
                im_tot += np.sum(images, axis=0)

        if self.m_stack == 'mean':
            im_stack = im_tot/float(nimages)
            self.m_image_out_port.set_all(im_stack[np.newaxis, ...])

        elif self.m_stack == 'median':
            im_stack = np.median(images, axis=0)
            self.m_image_out_port.set_all(im_stack[np.newaxis, ...])

        if self.m_derotate or self.m_stack is not None:
            self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.close_port()


class CombineTagsModule(ProcessingModule):
    """
    Pipeline module for combining tags from multiple database entries into a single tag.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tags: List[str],
                 image_out_tag: str,
                 check_attr: bool = True,
                 index_init: bool = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tags : list(str, )
            Tags of the database entries that are read as input and combined.
        image_out_tag : str
            Tag of the database entry that is written as output. Should not be present in
            *image_in_tags*.
        check_attr : bool
            Compare non-static attributes between the tags or combine all non-static attributes
            into the new database tag.
        index_init : bool
            Reinitialize the ``INDEX`` attribute. The frames are indexed in the order of tags names
            that are provided in *image_in_tags*.

        Returns
        -------
        NoneType
            None
        """

        super(CombineTagsModule, self).__init__(name_in=name_in)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        if image_out_tag in image_in_tags:
            raise ValueError('The \'image_out_tag\' should not be present in \'image_in_tags\'.')

        if len(image_in_tags) < 2:
            raise ValueError('The \'image_in_tags\' should contain at least two tags.')

        self.m_image_in_tags = image_in_tags
        self.m_check_attr = check_attr
        self.m_index_init = index_init

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Combines the frames of multiple tags into a single dataset
        and adds the static and non-static attributes. The values of the attributes are compared
        between the input tags to make sure that the input tags descent from the same data set.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute('MEMORY')

        image_in_port = []
        im_shape = []

        for i, item in enumerate(self.m_image_in_tags):
            image_in_port.append(self.add_input_port(item))
            im_shape.append(image_in_port[i].get_shape()[-2:])

        if len(set(im_shape)) > 1:
            raise ValueError('The size of the images should be the same for all datasets.')

        count = 0
        start_time = time.time()
        for i, item in enumerate(self.m_image_in_tags):
            progress(i, len(self.m_image_in_tags), 'Combining datasets...', start_time)

            nimages = image_in_port[i].get_shape()[0]
            frames = memory_frames(memory, nimages)

            for j, _ in enumerate(frames[:-1]):
                im_tmp = image_in_port[i][frames[j]:frames[j+1], ]
                self.m_image_out_port.append(im_tmp)

                if self.m_index_init:
                    index = np.arange(frames[j], frames[j+1], 1) + count

                    if i == 0 and j == 0:
                        self.m_image_out_port.add_attribute('INDEX', index, static=False)
                    else:
                        for ind in index:
                            self.m_image_out_port.append_attribute_data('INDEX', ind)

            static_attr = image_in_port[i].get_all_static_attributes()
            non_static_attr = image_in_port[i].get_all_non_static_attributes()

            for key in static_attr:
                status = self.m_image_out_port.check_static_attribute(key, static_attr[key])

                if status == 1:
                    self.m_image_out_port.add_attribute(key, static_attr[key], static=True)

                elif status == -1 and key[0:7] != 'History':
                    warnings.warn(f'The static keyword {key} is already used but with a different '
                                  f'value. It is advisable to only combine tags that descend from '
                                  f'the same data set.')

            for key in non_static_attr:
                values = image_in_port[i].get_attribute(key)
                status = self.m_image_out_port.check_non_static_attribute(key, values)

                if key != 'INDEX' or (key == 'INDEX' and not self.m_index_init):

                    if self.m_check_attr:
                        if key in ('PARANG', 'STAR_POSITION', 'INDEX', 'NFRAMES'):
                            if status == 1:
                                self.m_image_out_port.add_attribute(key, values, static=False)

                            else:
                                for j in values:
                                    self.m_image_out_port.append_attribute_data(key, j)

                        else:
                            if status == 1:
                                self.m_image_out_port.add_attribute(key, values, static=False)

                            if status == -1:
                                warnings.warn(f'The non-static keyword {key} is already used but '
                                              f'with different values. It is advisable to only '
                                              f'combine tags that descend from the same data set.')

                    else:
                        if status == 1:
                            self.m_image_out_port.add_attribute(key, values, static=False)

                        else:
                            for j in values:
                                self.m_image_out_port.append_attribute_data(key, j)

            count += nimages

        history = f'number of input tags = {np.size(self.m_image_in_tags)}'
        self.m_image_out_port.add_history('CombineTagsModule', history)
        self.m_image_out_port.close_port()
