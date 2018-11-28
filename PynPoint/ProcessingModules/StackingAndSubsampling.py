"""
Modules for stacking and subsampling of images.
"""

from __future__ import absolute_import

import sys
import warnings

import numpy as np

from six.moves import range

from PynPoint.Core.Processing import ProcessingModule
from PynPoint.Util.ModuleTools import progress, memory_frames
from PynPoint.Util.ImageTools import rotate_images


class StackAndSubsetModule(ProcessingModule):
    """
    Module for stacking subsets of images and/or selecting a random sample of images.
    """

    def __init__(self,
                 name_in="stacking_subset",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr",
                 random=None,
                 stacking=None):
        """
        Constructor of StackAndSubsetModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str
        :param random: Number of random images. All images are used if set to None.
        :type random: int
        :param stacking: Number of stacked images per subset. No stacking is applied if set
                         to None.
        :type stacking: int

        :return: None
        """

        super(StackAndSubsetModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_random = random
        self.m_stacking = stacking

        if self.m_stacking is None and self.m_random is None:
            warnings.warn("Both 'stacking' and 'random' are set to None.")

    def run(self):
        """
        Run method of the module. Stacks subsets of images and/or selects a random subset.

        :return: None
        """

        def _stack(nimages, im_shape, parang):
            im_new = None
            parang_new = None

            if self.m_stacking is not None:
                frames = memory_frames(self.m_stacking, nimages)

                nimages_new = np.size(frames)-1

                if parang is None:
                    parang_new = None
                else:
                    parang_new = np.zeros(nimages_new)

                im_new = np.zeros((nimages_new, im_shape[1], im_shape[2]))

                for i in range(nimages_new):
                    progress(i, nimages_new, "Running StackAndSubsetModule...")

                    if parang is not None:
                        parang_new[i] = np.mean(parang[frames[i]:frames[i+1]])

                    im_new[i, ] = np.mean(self.m_image_in_port[frames[i]:frames[i+1], ], axis=0)

                im_shape = im_new.shape

            else:
                if parang is not None:
                    parang_new = np.copy(parang)

            return im_shape, im_new, parang_new

        def _subset(im_shape, im_new, parang_new):
            if self.m_random is not None:
                choice = np.random.choice(im_shape[0], self.m_random, replace=False)
                choice = np.sort(choice)

                if parang_new is None:
                    parang_new = None
                else:
                    parang_new = parang_new[choice]

                if self.m_stacking is None:
                    im_new = self.m_image_in_port[choice, ]
                else:
                    im_new = im_new[choice, ]

            if im_new.ndim == 2:
                nimages = 1
            elif im_new.ndim == 3:
                nimages = im_new.shape[0]

            return nimages, im_new, parang_new

        non_static = self.m_image_in_port.get_all_non_static_attributes()

        im_shape = self.m_image_in_port.get_shape()
        nimages = im_shape[0]

        if self.m_random is not None:
            if self.m_stacking is None and im_shape[0] < self.m_random:
                raise ValueError("The number of images of the destination subset is larger than " \
                                 "the number of images in the source.")

            elif self.m_stacking is not None and \
                        int(float(im_shape[0])/float(self.m_stacking)) < self.m_random:
                raise ValueError("The number of images of the destination subset is larger than " \
                                 "the number of images in the stacked source.")

        if "PARANG" in non_static:
            parang = self.m_image_in_port.get_attribute("PARANG")
        else:
            parang = None

        im_shape, im_new, parang_new = _stack(nimages, im_shape, parang)
        nimages, im_new, parang_new = _subset(im_shape, im_new, parang_new)

        sys.stdout.write("Running StackAndSubsetModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.set_all(im_new, keep_attributes=True)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_attribute("INDEX", np.arange(0, nimages, 1), static=False)

        if parang is not None:
            self.m_image_out_port.add_attribute("PARANG", parang_new, static=False)

        if "NFRAMES" in non_static:
            self.m_image_out_port.del_attribute("NFRAMES")

        self.m_image_out_port.add_history_information("Stack and subset",
                                                      "stacking ="+str(self.m_stacking)+
                                                      "random ="+str(self.m_random))

        self.m_image_out_port.close_port()


class MeanCubeModule(ProcessingModule):
    """
    Module for calculating the mean of each individual cube associated with a database tag.
    """

    def __init__(self,
                 name_in="mean_cube",
                 image_in_tag="im_arr",
                 image_out_tag="im_mean"):
        """
        Constructor of MeanCubeModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry with the mean collapsed images that are
                              written as output. Should be different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(MeanCubeModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Uses the NFRAMES attribute to select the images of each cube,
        calculates the mean of each cube, and saves the data and attributes.

        :return: None
        """

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        non_static = self.m_image_in_port.get_all_non_static_attributes()

        nframes = self.m_image_in_port.get_attribute("NFRAMES")

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        current = 0

        for i, frames in enumerate(nframes):
            progress(i, len(nframes), "Running MeanCubeModule...")

            mean_frame = np.mean(self.m_image_in_port[current:current+frames, ],
                                 axis=0)

            self.m_image_out_port.append(mean_frame,
                                         data_dim=3)

            current += frames

        sys.stdout.write("Running MeanCubeModule... [DONE]\n")
        sys.stdout.flush()

        nimages = np.size(nframes)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        if "INDEX" in non_static:
            index = np.arange(0, nimages, 1, dtype=np.int)
            self.m_image_out_port.add_attribute("INDEX", index, static=False)

        if "NFRAMES" in non_static:
            nframes = np.ones(nimages, dtype=np.int)
            self.m_image_out_port.add_attribute("NFRAMES", nframes, static=False)

        self.m_image_out_port.close_port()


class DerotateAndStackModule(ProcessingModule):
    """
    Module for derotating and/or stacking (i.e., taking the median or average) of the images.
    """

    def __init__(self,
                 name_in="derotate_stack",
                 image_in_tag="im_arr",
                 image_out_tag="im_stack",
                 derotate=True,
                 stack=None,
                 extra_rot=0.):
        """
        Constructor of DerotateAndStackModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. The output is
                              either 2D (*stack=False*) or 3D (*stack=True*).
        :type image_out_tag: str
        :param derotate: Derotate the images with the PARANG attribute.
        :type derotate: bool
        :param stack: Type of stacking applied after optional derotation ("mean", "median",
                      or None for no stacking).
        :type stack: str
        :param extra_rot: Additional rotation angle of the images in clockwise direction (deg).
        :type extra_rot: float

        :return: None
        """

        super(DerotateAndStackModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_derotate = derotate
        self.m_stack = stack
        self.m_extra_rot = extra_rot

    def run(self):
        """
        Run method of the module. Uses the PARANG attributes to derotate the images (if *derotate*
        is set to True) and applies an optional mean or median stacking afterwards.

        :return: None
        """

        def _initialize(ndim, npix):
            if ndim == 2:
                nimages = 1
            elif ndim == 3:
                nimages = self.m_image_in_port.get_shape()[0]

            if self.m_stack == "median":
                frames = [0, nimages]
            else:
                frames = memory_frames(memory, nimages)

            if self.m_stack == "mean":
                im_tot = np.zeros((npix, npix))
            else:
                im_tot = None

            return nimages, frames, im_tot

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        memory = self._m_config_port.get_attribute("MEMORY")

        if self.m_derotate:
            parang = self.m_image_in_port.get_attribute("PARANG")

        ndim = self.m_image_in_port.get_ndim()
        npix = self.m_image_in_port.get_shape()[1]

        nimages, frames, im_tot = _initialize(ndim, npix)

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running DerotateAndStackModule...")

            images = self.m_image_in_port[frames[i]:frames[i+1], ]

            if self.m_derotate:
                angles = -parang[frames[i]:frames[i+1]]+self.m_extra_rot
                images = rotate_images(images, angles)

            if self.m_stack is None:
                if ndim == 2:
                    self.m_image_out_port.set_all(images)
                elif ndim == 3:
                    self.m_image_out_port.append(images, data_dim=3)

            elif self.m_stack == "mean":
                im_tot += np.sum(images, axis=0)

        sys.stdout.write("Running DerotateAndStackModule... [DONE]\n")
        sys.stdout.flush()

        if self.m_stack == "mean":
            self.m_image_out_port.set_all(im_tot/float(nimages))

        elif self.m_stack == "median":
            self.m_image_out_port.set_all(np.median(images, axis=0))

        if self.m_derotate or self.m_stack is not None:
            self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class CombineTagsModule(ProcessingModule):
    """
    Module for combining tags from multiple database entries into a single tag.
    """

    def __init__(self,
                 image_in_tags,
                 check_attr=True,
                 index_init=False,
                 name_in="combine_tags",
                 image_out_tag="im_arr_combined"):
        """
        Constructor of CombineTagsModule.

        :param image_in_tags: Tags of the database entries that are read as input and combined.
        :type image_in_tags: (str, str, )
        :param check_attr: Compare non-static attributes between the tags or combine all non-static
                           attributes into the new database tag.
        :type check_attr: bool
        :param index_init: Reinitialize the INDEX attribute. Index frames in the order of the input
                           tags.
        :type: index_init: bool
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_out_tag: Tag of the database entry that is written as output. Should not be
                              present in *image_in_tags*.
        :type image_out_tag: str

        :return: None
        """

        super(CombineTagsModule, self).__init__(name_in=name_in)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        if image_out_tag in image_in_tags:
            raise ValueError("The name of 'image_out_tag' can not be present in 'image_in_tags'.")

        if len(image_in_tags) < 2:
            raise ValueError("The 'image_in_tags' should contain at least two tags.")

        self.m_image_in_tags = image_in_tags
        self.m_check_attr = check_attr
        self.m_index_init = index_init

    def run(self):
        """
        Run method of the module. Combines the frames of multiple tags into a single output tag
        and adds the static and non-static attributes. The values of the attributes are compared
        between the input tags to make sure that the input tags descent from the same data set.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")

        count = 0
        for i, item in enumerate(self.m_image_in_tags):
            progress(i, len(self.m_image_in_tags), "Running CombineTagsModule...")

            image_in_port = self.add_input_port(item)
            nimages = image_in_port.get_shape()[0]

            frames = memory_frames(memory, nimages)

            for j, _ in enumerate(frames[:-1]):
                im_tmp = image_in_port[frames[j]:frames[j+1], ]
                self.m_image_out_port.append(im_tmp)

                if self.m_index_init:
                    index = np.arange(frames[j], frames[j+1], 1) + count

                    if i == 0 and j == 0:
                        self.m_image_out_port.add_attribute("INDEX", index, static=False)
                    else:
                        for ind in index:
                            self.m_image_out_port.append_attribute_data("INDEX", ind)

            static_attr = image_in_port.get_all_static_attributes()
            non_static_attr = image_in_port.get_all_non_static_attributes()

            for key in static_attr:
                status = self.m_image_out_port.check_static_attribute(key, static_attr[key])

                if status == 1:
                    self.m_image_out_port.add_attribute(key, static_attr[key], static=True)

                elif status == -1 and key[0:7] != "History":
                    warnings.warn('The static keyword %s is already used but with a different '
                                  'value. It is advisable to only combine tags that descend from '
                                  'the same data set.' % key)

            for key in non_static_attr:
                values = image_in_port.get_attribute(key)
                status = self.m_image_out_port.check_non_static_attribute(key, values)

                if key != "INDEX" or (key == "INDEX" and not self.m_index_init):

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
                                warnings.warn('The non-static keyword %s is already used but with '
                                              'different values. It is advisable to only combine '
                                              'tags that descend from the same data set.' % key)

                    else:
                        if status == 1:
                            self.m_image_out_port.add_attribute(key, values, static=False)

                        else:
                            for j in values:
                                self.m_image_out_port.append_attribute_data(key, j)

            count += nimages

        sys.stdout.write("Running CombineTagsModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.add_history_information("Database entries combined",
                                                      str(np.size(self.m_image_in_tags)))

        self.m_image_out_port.close_port()
