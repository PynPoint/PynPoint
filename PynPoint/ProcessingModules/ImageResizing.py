"""
Modules with simple pre-processing tools.
"""

import sys
import warnings

import numpy as np

from skimage.transform import rescale

from PynPoint.Util.Progress import progress
from PynPoint.Core.Processing import ProcessingModule


class CropImagesModule(ProcessingModule):
    """
    Module for cropping of images around a given position.
    """

    def __init__(self,
                 size,
                 center=None,
                 name_in="crop_image",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_cropped"):
        """
        Constructor of CropImagesModule.

        :param size: New image size (arcsec). The same size will be used for both image dimensions.
        :type size: float
        :param center: Tuple (x0, y0) with the new image center. Python indexing starts at 0. The
                       center of the input images will be used when *center* is set to *None*.
        :type center: tuple, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(CropImagesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_size = size
        self.m_center = center

    def run(self):
        """
        Run method of the module. Reduces the image size by cropping around an given position.

        :return: None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_size = int(self.m_size/pixscale)

        def image_cutting(image_in,
                          size,
                          center):

            if center is None:
                x_off = (image_in.shape[0] - size) / 2
                y_off = (image_in.shape[1] - size) / 2

                if size > image_in.shape[0] or size > image_in.shape[1]:
                    raise ValueError("Input frame resolution smaller than target image resolution.")

                image_out = image_in[y_off:y_off+size, x_off:x_off+size]

            else:
                x_in = int(center[0] - size/2)
                y_in = int(center[1] - size/2)

                x_out = int(center[0] + size/2)
                y_out = int(center[1] + size/2)

                if x_in < 0 or y_in < 0 or x_out > image_in.shape[0] or y_out > image_in.shape[1]:
                    raise ValueError("Target image resolution does not fit inside the input frame "
                                     "resolution.")

                image_out = image_in[y_in:y_out, x_in:x_out]

            return image_out

        self.apply_function_to_images(image_cutting,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running CropImagesModule...",
                                      func_args=(self.m_size, self.m_center),
                                      num_images_in_memory=memory)

        self.m_image_out_port.add_history_information("Image cropped", str(self.m_size))
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_database()


class ScaleImagesModule(ProcessingModule):
    """
    Module for rescaling of an image.
    """

    def __init__(self,
                 scaling_dim,
                 scaling_flux=1.,
                 name_in="scaling",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_scaled"):
        """
        Constructor of ScaleImagesModule.

        :param scaling_dim: Scaling factor for upsampling (*scaling_factor* > 1) and downsampling
                        (0 < *scaling_factor* < 1).
        :type scaling_dim: float
        :param scaling_flux: Scaling factor for the total image flux
        :type scaling_flux: float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(ScaleImagesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_scaling_dim = scaling_dim
        self.m_scaling_flux = scaling_flux

    def run(self):
        """
        Run method of the module. Rescales an image with a fifth order spline interpolation and a
        reflecting boundary condition.

        :return: None
        """

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        def image_scaling(image_in,
                          scaling_dim,
                          scaling_flux):

            sum_before = np.sum(image_in)
            tmp_image = rescale(image=np.asarray(image_in, dtype=np.float64),
                                scale=(scaling_dim, scaling_dim),
                                order=5,
                                mode="reflect")

            sum_after = np.sum(tmp_image)
            return tmp_image * (sum_before / sum_after) * scaling_flux

        self.apply_function_to_images(image_scaling,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running ScaleImagesModule...",
                                      func_args=(self.m_scaling_dim,self.m_scaling_flux),
                                      num_images_in_memory=memory)

        self.m_image_out_port.add_history_information("Images scaled", "dim = "+str(self.m_scaling_dim)+", flux = "+str(self.m_scaling_flux))
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_attribute("PIXSCALE", pixscale/self.m_scaling_dim)
        self.m_image_out_port.close_database()


class AddLinesModule(ProcessingModule):
    """
    Module to add lines of pixels to increase the size of an image.
    """

    def __init__(self,
                 lines,
                 name_in="add_lines",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_add"):
        """
        Constructor of AddLinesModule.

        :param lines: Tuple with the number of additional lines in left, right, bottom, and top
                      direction.
        :type lines: tuple, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag* unless *MEMORY* is set to *None*.
        :type image_out_tag: str

        :return: None
        """

        super(AddLinesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_lines = lines

    def run(self):
        """
        Run method of the module. Adds lines of zero-value pixels to increase the size of an image.

        :return: None
        """

        memory = self._m_config_port.get_attribute("MEMORY")
        shape_in = self.m_image_in_port.get_shape()

        if np.size(shape_in) != 3:
            raise ValueError("Expecting a 3D array with images.")

        if any(np.asarray(self.m_lines) < 0.):
            raise ValueError("The lines argument should contain values equal to or larger than "
                             "zero.")

        shape_out = (shape_in[1]+int(self.m_lines[2])+int(self.m_lines[3]),
                     shape_in[2]+int(self.m_lines[0])+int(self.m_lines[1]))

        if shape_out[0] != shape_out[1]:
            warnings.warn("The dimensions of the output images %s are not equal. PynPoint only "
                          "supports square images." % str(shape_out))

        def add_lines(image_in):
            image_out = np.zeros(shape_out)
            image_out[int(self.m_lines[2]):-int(self.m_lines[3]),
                      int(self.m_lines[0]):-int(self.m_lines[1])] = image_in

            return image_out

        self.apply_function_to_images(add_lines,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running AddLinesModule...",
                                      num_images_in_memory=memory)

        self.m_image_out_port.add_history_information("Lines added", str(self.m_lines))
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_database()


class RemoveLinesModule(ProcessingModule):
    """
    Module to decrease the dimensions of an image by removing lines of pixels.
    """

    def __init__(self,
                 lines,
                 name_in="cut_top",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_cut"):
        """
        Constructor of RemoveLinesModule.

        :param lines: Tuple with the number of lines to be removed in left, right, bottom,
                      and top direction.
        :type lines: tuple, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(RemoveLinesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_lines = lines

    def run(self):
        """
        Run method of the module. Removes the top *num_lines* lines from each frame.

        :return: None
        """

        memory = self._m_config_port.get_attribute("MEMORY")

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output tags should be different.")

        def remove_lines(image_in):
            shape_in = image_in.shape
            return image_in[int(self.m_lines[2]):shape_in[0]-int(self.m_lines[3]),
                            int(self.m_lines[0]):shape_in[1]-int(self.m_lines[1])]

        self.apply_function_to_images(remove_lines,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running RemoveLinesModule...",
                                      num_images_in_memory=memory)

        self.m_image_out_port.add_history_information("Lines removed", str(self.m_lines))
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_database()


class CombineTagsModule(ProcessingModule):
    """
    Module for combining tags from multiple database entries into a single tag.
    """

    def __init__(self,
                 image_in_tags,
                 check_attr=True,
                 name_in="combine_tags",
                 image_out_tag="im_arr_combined"):
        """
        Constructor of CombineTagsModule.

        :param image_in_tags: Tags of the database entries that are combined.
        :type image_in_tags: tuple, str
        :param check_attr: Compare non-static attributes between the tags or combine all non-static
                           attributes into the new database tag.
        :type check_attr: bool
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
            raise ValueError("The name of image_out_tag can not be present in image_in_tags.")

        self.m_image_in_tags = image_in_tags
        self.m_check_attr = check_attr

    def run(self):
        """
        Run method of the module. Combines the frames of multiple tags into a single output tag
        and adds the static and non-static attributes. The values of the attributes are compared
        between the input tags to make sure that the input tags decent from the same data set.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if len(self.m_image_in_tags) < 2:
            raise ValueError("The tuple of image_in_tags should contain at least two tags.")

        memory = self._m_config_port.get_attribute("MEMORY")

        for i, item in enumerate(self.m_image_in_tags):
            progress(i, len(self.m_image_in_tags), "Running CombineTagsModule...")

            image_in_port = self.add_input_port(item)

            num_frames = image_in_port.get_shape()[0]
            num_stacks = int(float(num_frames)/float(memory))

            for j in range(num_stacks):
                frame_start = j*memory
                frame_end = j*memory+memory

                im_tmp = image_in_port[frame_start:frame_end, ]
                self.m_image_out_port.append(im_tmp)

            if num_frames%memory > 0:
                frame_start = num_stacks*memory
                frame_end = num_frames

                im_tmp = image_in_port[frame_start:frame_end, ]
                self.m_image_out_port.append(im_tmp)

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

                if self.m_check_attr:
                    if key == "NFRAMES" or key == "PARANG" or key == "STAR_POSITION":
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
                                          'different values. It is advisable to only combine tags '
                                          'that descend from the same data set.' % key)

                else:
                    if status == 1:
                        self.m_image_out_port.add_attribute(key, values, static=False)
                    else:
                        for j in values:
                            self.m_image_out_port.append_attribute_data(key, j)

        sys.stdout.write("Running CombineTagsModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.add_history_information("Database entries combined",
                                                      str(np.size(self.m_image_in_tags)))

        self.m_image_out_port.close_database()
