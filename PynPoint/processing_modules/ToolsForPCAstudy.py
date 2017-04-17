from PynPoint.core import ProcessingModule

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage


class RemoveMeanOrMedianModule(ProcessingModule):

    def __init__(self,
                 mode="mean",
                 name_in="remove_mean",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_no_mean",
                 number_of_images_in_memory=100):

        super(RemoveMeanOrMedianModule, self).__init__(name_in=name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_mode = mode

    def run(self):

        def image_scaling(image_in,
                          sub_img_in):

            return image_in - sub_img_in

        if self.m_mode == "mean":
            sub_img = np.mean(self.m_image_in_port.get_all(), axis=0)

        else:
            sub_img = np.median(self.m_image_in_port.get_all(), axis=0)

        self.apply_function_to_images(image_scaling,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(sub_img,),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        if self.m_mode == "mean":
            self.m_image_out_port.add_history_information("Subtracted",
                                                          "Mean")
        else:
            self.m_image_out_port.add_history_information("Subtracted",
                                                          "Median")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class RotateFramesModule(ProcessingModule):

    def __init__(self,
                 mode="normal",
                 name_in="rotation",
                 image_in_tag="im_arr",
                 rot_out_tag="im_arr_rot"):

        super(RotateFramesModule, self).__init__(name_in=name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_rot_out_port = self.add_output_port(rot_out_tag)

        self.m_mode = mode

    def run(self):

        para_angles = self.m_image_in_port.get_attribute("NEW_PARA")

        if self.m_mode == "normal":
            delta_para = - para_angles
        else:
            delta_para = para_angles

        im_data = self.m_image_in_port.get_all()

        res_rot = np.zeros(shape=im_data.shape)
        for i in range(0, len(delta_para)):
            res_temp = im_data[i, ]

            res_rot[i, ] = ndimage.rotate(res_temp,
                                          delta_para[i],
                                          reshape=False,
                                          order=0)

        self.m_rot_out_port.set_all(res_rot)
        self.m_rot_out_port.add_history_information("Rotaion Mode",
                                                    self.m_mode)

        self.m_rot_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_rot_out_port.close_port()


class CombineADIModule(ProcessingModule):

    def __init__(self,
                 type_in="mean",
                 name_in="combine",
                 image_in_tag="im_arr",
                 image_out_tag="im_ADI"):

        super(CombineADIModule, self).__init__(name_in=name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_type = type_in

    def run(self):

        im_data = self.m_image_in_port.get_all()

        if self.m_type == "mean":
            self.m_image_out_port.set_all(np.mean(im_data, axis=0))
            self.m_image_out_port.add_history_information("ADI combination",
                                                          "mean")
        else:
            self.m_image_out_port.set_all(np.median(im_data, axis=0))
            self.m_image_out_port.add_history_information("ADI combination",
                                                          "median")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class SimpleSpeckleSubtraction(ProcessingModule):

    def __init__(self,
                 name_in="speckle_subtraction",
                 image_in_tag="im_arr",
                 speckle_in_tag="speckle_in",
                 image_out_tag="im_planet"):

        super(SimpleSpeckleSubtraction, self).__init__(name_in=name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_speckle_in_port = self.add_input_port(speckle_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        im_data = self.m_image_in_port.get_all()
        speckle_data = self.m_speckle_in_port.get_all()

        self.m_image_out_port.set_all(im_data - speckle_data)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()