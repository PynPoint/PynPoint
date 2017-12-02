"""
Modules with tools for frame selection.
"""

from PynPoint.core import ProcessingModule

import numpy as np

class RemoveFramesModule(ProcessingModule):
    """
    Module for removing frames.
    """

    def __init__(self,
                 frame_indices,
                 name_in="remove_frames",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_remove",
                 num_image_in_memory=100):
        """
        Constructor of RemoveFramesModule.

        :param frame_indices: Frame indices to be removed. Python indexing starts at 0.
        :type frame_indices: tuple or array, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str
        :param num_image_in_memory: Number of frames that are simultaneously loaded into the memory.
        :type num_image_in_memory: int
        :return: None
        """

        super(RemoveFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_frame_indices = np.asarray(frame_indices)
        self.m_image_memory = num_image_in_memory

    def run(self):
        """
        Run method of the module. Removes the frames, removes the associated NEW_PARA values,
        updates the NAXIS3 value, and saves the data and attributes.

        :return: None
        """

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        if np.size(np.where(self.m_frame_indices >= self.m_image_in_port.get_shape()[0])) > 0:
            raise ValueError("Some values in frame_indices are larger than the total number of "
                             "available frames, %s." % str(self.m_image_in_port.get_shape()[0]))

        if "NEW_PARA" not in self.m_image_in_port.get_all_non_static_attributes():
            raise ValueError("NEW_PARA not found in header. Parallactic angles should be "
                             "provided for all frames before any frames can be removed.")

        num_subsets = int(self.m_image_in_port.get_shape()[0]/self.m_image_memory)

        # Reading subsets of num_image_in_memory frames and remove frame_indices

        for i in range(num_subsets):

            tmp_im = self.m_image_in_port[i*self.m_image_memory:(i+1)*self.m_image_memory, :, :]

            index_del = np.where(np.logical_and(self.m_frame_indices >= i*self.m_image_memory, \
                                 self.m_frame_indices < (i+1)*self.m_image_memory))

            if np.size(index_del) > 0:
                tmp_im = np.delete(tmp_im,
                                   self.m_frame_indices[index_del]%self.m_image_memory,
                                   axis=0)

            if i == 0:
                self.m_image_out_port.set_all(tmp_im, keep_attributes=True)
            else:
                self.m_image_out_port.append(tmp_im)

        # Adding the leftover frames that do not fit in an integer amount of num_image_in_memory

        index_del = np.where(self.m_frame_indices >= num_subsets*self.m_image_memory)

        tmp_im = self.m_image_in_port[num_subsets*self.m_image_memory: \
                                      self.m_image_in_port.get_shape()[0], :, :]

        tmp_im = np.delete(tmp_im,
                           self.m_frame_indices[index_del]%self.m_image_memory,
                           axis=0)

        self.m_image_out_port.append(tmp_im)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        # Update parallactic angles

        par_in = self.m_image_in_port.get_attribute("NEW_PARA")
        par_out = np.delete(par_in,
                            self.m_frame_indices)

        self.m_image_out_port.add_attribute("NEW_PARA", par_out, static=False)

        # Update cube sizes

        size_in = self.m_image_in_port.get_attribute("NAXIS3")
        size_out = np.copy(size_in)

        num_frames = 0
        for i, frames in enumerate(size_in):
            index_del = np.where(np.logical_and(self.m_frame_indices >= num_frames, \
                                 self.m_frame_indices < num_frames+frames))

            size_out[i] -= np.size(index_del)

            num_frames += frames

        self.m_image_out_port.add_attribute("NAXIS3", size_out, static=False)

        self.m_image_out_port.add_history_information("Removed frames",
                                                      str(np.size(self.m_frame_indices)))

        # Update star position (if present)

        if "STAR_POSITION_X" in self.m_image_in_port.get_all_non_static_attributes():

            starpos_in = self.m_image_in_port.get_attribute("STAR_POSITION_X")
            starpos_out = np.delete(starpos_in,
                                    self.m_frame_indices,
                                    axis=0)

            self.m_image_out_port.add_attribute("STAR_POSITION_X", starpos_out, static=False)

        if "STAR_POSITION_Y" in self.m_image_in_port.get_all_non_static_attributes():

            starpos_in = self.m_image_in_port.get_attribute("STAR_POSITION_Y")
            starpos_out = np.delete(starpos_in,
                                    self.m_frame_indices,
                                    axis=0)

            self.m_image_out_port.add_attribute("STAR_POSITION_Y", starpos_out, static=False)

        self.m_image_out_port.close_port()
