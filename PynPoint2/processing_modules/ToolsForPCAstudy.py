from PynPoint2.core import ProcessingModule

import numpy as np

from scipy import ndimage
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV


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
                                          order=5)

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


class ComputeModeModule(ProcessingModule):
    def __init__(self,
                 cross_validation_space=np.linspace(0.1, 1.0, 30),
                 cross_validation_fold=10,
                 search_space=np.linspace(-100, 100, 1000),
                 name_in="mode_calculation_module",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_mode",
                 number_of_rows_in_memory=100):

        super(ComputeModeModule, self).__init__(name_in=name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_rows_in_memory = number_of_rows_in_memory
        self.m_cross_validation_space = cross_validation_space
        self.m_cross_validation_fold = cross_validation_fold
        self.m_search_space = search_space

    def run(self):

        def calculate_mode_line(line_in):
            # estimate bandwidth
            '''
            grid = GridSearchCV(KernelDensity(),
                                {'bandwidth': self.m_cross_validation_space},
                                cv=self.m_cross_validation_fold)  # cross-validation
            grid.fit(line_in[:, None])
            bandwidth = grid.best_params_['bandwidth']'''

            # KDE
            kde_skl = KernelDensity(bandwidth=2.0)
            kde_skl.fit(line_in[:, np.newaxis])
            # score_samples() returns the log-likelihood of the samples
            log_pdf = kde_skl.score_samples(self.m_search_space[:, np.newaxis])

            print np.array([self.m_search_space[np.argsort(log_pdf)[-1]]])

            return np.array([self.m_search_space[np.argsort(log_pdf)[-1]]])

        self.apply_function_to_line_in_time_multi_processing(calculate_mode_line,
                                                             self.m_image_in_port,
                                                             self.m_image_out_port,
                                                             num_rows_in_memory=
                                                             self.m_number_of_rows_in_memory)

        self.m_image_out_port.add_history_information("Mode Estimate", "KDE + " \
                                                      + str(self.m_cross_validation_fold) + \
                                                      " fold cross-validation")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()
