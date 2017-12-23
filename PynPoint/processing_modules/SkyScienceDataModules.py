"""
Modules for subtraction of the sky background.
"""

import numpy as np
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift

from PynPoint.core import ProcessingModule


class MeanSkyCubes(ProcessingModule):
    """
    Module for calculating the mean sky background for each data cube of sky frames.
    """

    def __init__(self,
                 name_in="mean_sky_frames",
                 sky_in_tag="sky_raw_arr",
                 sky_out_tag="sky_arr"):
        """
        Constructor of MeanSkyCubes.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param sky_in_tag: Tag of the database entry with sky frames that is read as input.
        :type sky_in_tag: str
        :param sky_out_tag: Tag of the database entry with the mean sky frames that is written
                            as output. Should be different from *sky_in_tag*.
        :type sky_out_tag: str
        :return: None
        """

        super(MeanSkyCubes, self).__init__(name_in=name_in)

        self.m_sky_in_port = self.add_input_port(sky_in_tag)
        self.m_sky_out_port = self.add_output_port(sky_out_tag)

    def run(self):
        """
        Run method of the module. Selects the frames of each cube of sky images, calculates the
        corresponding mean sky background, and saves the data and attributes.

        :return: None
        """

        if self.m_sky_out_port.tag == self.m_sky_in_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        list_of_frame_numbers = self.m_sky_in_port.get_attribute("NFRAMES")

        self.m_sky_out_port.del_all_data()
        self.m_sky_out_port.del_all_attributes()

        current_frame = 0

        for frame_number in list_of_frame_numbers:

            mean_frame = np.mean(self.m_sky_in_port[current_frame:current_frame+frame_number, :, :],
                                 axis=0)

            current_frame += frame_number

            self.m_sky_out_port.append(mean_frame,
                                       data_dim=3)

        self.m_sky_out_port.copy_attributes_from_input_port(self.m_sky_in_port)
        self.m_sky_out_port.close_port()


class SkySubtraction(ProcessingModule):
    """
    Module for sky subtraction.
    """

    def __init__(self,
                 name_in="sky_subtraction",
                 sky_in_tag="sky_arr",
                 science_data_in_tag="im_arr",
                 science_data_out_tag="im_arr",
                 mode="next"):
        """
        Constructor of SkySubtraction.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param sky_in_tag: Tag of the database entry with sky frames that is read as input.
        :type sky_in_tag: str
        :param science_data_in_tag: Tag of the database entry with science frames that is read
                                    as input.
        :type science_data_in_tag: str
        :param science_data_out_tag: Tag of the database entry that is written as output.
        :type science_data_out_tag: str
        :return: None
        """

        super(SkySubtraction, self).__init__(name_in=name_in)

        self.m_sky_in_port = self.add_input_port(sky_in_tag)
        self.m_science_in_port = self.add_input_port(science_data_in_tag)
        self.m_science_out_port = self.add_output_port(science_data_out_tag)

        self.m_time_stamps = []

        if mode in ["next", "previous", "both"]:
            self.m_mode = mode
        else:
            raise ValueError("Mode needs to be next, previous or both.")

    def _create_time_stamp_list(self):
        """
        Internal method for assigning a time stamp, based on the exposure number ID, to each cube
        of sky and science frames.
        """

        class TimeStamp:
            def __init__(self,
                         time,
                         sky_or_science,
                         index):
                self.m_time = time
                self.m_sky_or_science = sky_or_science
                self.m_index = index

            def __repr__(self):
                return repr((self.m_time,
                             self.m_sky_or_science,
                             self.m_index))

        # add time stamps of Sky data
        dates = self.m_sky_in_port.get_attribute("EXP_NO")

        for i, item in enumerate(dates):
            self.m_time_stamps.append(TimeStamp(item,
                                                "SKY",
                                                i))

        # add time stamps of Science data
        dates = self.m_science_in_port.get_attribute("EXP_NO")
        num_frames = self.m_science_in_port.get_attribute("NFRAMES")

        frames_count = 0
        for i, item in enumerate(dates):
            self.m_time_stamps.append(TimeStamp(item,
                                                "SCIENCE",
                                                slice(frames_count, frames_count+num_frames[i])))
            frames_count += num_frames[i]

        self.m_time_stamps = sorted(self.m_time_stamps, key=lambda time_stamp: time_stamp.m_time)

    def calc_sky_frame(self,
                       index_of_science_data):
        """
        Method for finding the required sky frame (next, previous, or the mean of next and
        previous) by comparing the time stamp of the science frame with preceding and following
        sky frames.
        """

        # check if there is at least one SKY in the database
        if not any(x.m_sky_or_science == "SKY" for x in self.m_time_stamps):
            raise ValueError('List of time stamps does not contain any SKY frames')

        def search_for_next_sky():
            for i in range(index_of_science_data, len(self.m_time_stamps)):
                if self.m_time_stamps[i].m_sky_or_science == "SKY":
                    return self.m_sky_in_port[self.m_time_stamps[i].m_index, :, :]

            # no next sky found look for previous sky
            return search_for_previous_sky()

        def search_for_previous_sky():
            for i in reversed(range(0, index_of_science_data)):
                if self.m_time_stamps[i].m_sky_or_science == "SKY":
                    return self.m_sky_in_port[self.m_time_stamps[i].m_index, :, :]

            # no previous sky found look for next sky
            return search_for_next_sky()

        if self.m_mode == "next":
            return search_for_next_sky()

        if self.m_mode == "previous":
            return search_for_previous_sky()

        if self.m_mode == "both":
            previous_sky = search_for_previous_sky()
            next_sky = search_for_next_sky()
            return (previous_sky + next_sky)/2.0

    def run(self):
        """
        Run method of the module. Create time stamp list, get sky and science frames, and subtract
        the sky background from the science frames.

        :return: None
        """

        self._create_time_stamp_list()

        self.m_science_out_port.del_all_data()
        self.m_science_out_port.del_all_attributes()

        # Number of slices with science frames
        num_science = sum(s.m_sky_or_science == "SCIENCE" for s in self.m_time_stamps)

        sky_count = 0

        for i, time_entry in enumerate(self.m_time_stamps):

            if time_entry.m_sky_or_science == "SCIENCE":
                print "Subtract background from file " + str(i+1-sky_count) + " of " + \
                      str(num_science) + " files..."

            elif time_entry.m_sky_or_science == "SKY":
                sky_count += 1
                continue

            # get sky image
            sky = self.calc_sky_frame(i)

            # get science data
            science = self.m_science_in_port[time_entry.m_index, :, :]

            self.m_science_out_port.append(science - sky[None, :, :],
                                           data_dim=3)

        self.m_science_out_port.copy_attributes_from_input_port(self.m_science_in_port)
        self.m_science_out_port.add_history_information("background_subtraction",
                                                        "using Sky Flat subtraction")
        self.m_science_out_port.close_port()


class AlignmentSkyAndScienceDataModule(ProcessingModule):

    def __init__(self,
                 position_of_center,
                 name_in="align_sky_and_science",
                 science_in_tag="science_arr",
                 sky_in_tag="sky_arr",
                 science_out_tag="science_arr",
                 sky_out_tag="sky_arr",
                 interpolation="spline",
                 size_of_center=(100, 100),
                 accuracy=10):

        super(AlignmentSkyAndScienceDataModule, self).__init__(name_in)

        # Ports

        self.m_science_in_port = self.add_input_port(science_in_tag)
        self.m_science_out_port = self.add_output_port(science_out_tag)

        self.m_sky_in_port = self.add_input_port(sky_in_tag)
        self.m_sky_out_port = self.add_output_port(sky_out_tag)

        # Parameter
        self.m_interpolation = interpolation
        self.m_accuracy = accuracy
        self.m_center_size = size_of_center
        self.m_center_position = position_of_center
        self.m_x_off = self.m_center_position[0] - (self.m_center_size[0] / 2)
        self.m_y_off = self.m_center_position[1] - (self.m_center_size[1] / 2)

    def run(self):

        self.m_num_images_in_memory = self._m_config_port.get_attribute("MEMORY")

        def cut_image_around_position(image_in):
            return image_in[self.m_x_off: self.m_center_size[0] + self.m_x_off,
                            self.m_y_off:self.m_center_size[1] + self.m_y_off]

        # create reference image
        ref_image = cut_image_around_position(self.m_sky_in_port[0])

        def align_single_image(image_in):

            offset, _, _ = register_translation(ref_image,
                                                cut_image_around_position(image_in),
                                                self.m_accuracy)

            if self.m_interpolation == "spline":
                tmp_image = shift(image_in, offset, order=5)

            elif self.m_interpolation == "fft":
                tmp_image_spec = fourier_shift(np.fft.fftn(image_in), offset)
                tmp_image = np.fft.ifftn(tmp_image_spec)

            elif self.m_interpolation == "bilinear":
                tmp_image = shift(image_in, offset, order=1)

            else:
                raise ValueError("Interpolation needs to be spline, bilinear or fft")

            return tmp_image

        # align all Science data
        self.apply_function_to_images(align_single_image,
                                      self.m_science_in_port,
                                      self.m_science_out_port,
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_science_out_port.copy_attributes_from_input_port(self.m_science_in_port)

        history = "cross-correlation"
        self.m_science_out_port.add_history_information("Sky-Science alignment",
                                                        history)

        # align all Sky data
        self.apply_function_to_images(align_single_image,
                                      self.m_sky_in_port,
                                      self.m_sky_out_port,
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_sky_out_port.copy_attributes_from_input_port(self.m_sky_in_port)

        history = "cross-correlation"
        self.m_sky_out_port.add_history_information("Sky-Science alignment",
                                                    history)

        self.m_sky_out_port.close_port()
