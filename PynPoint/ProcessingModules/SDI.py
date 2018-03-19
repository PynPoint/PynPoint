"""
Modules with tools for preparation to SDI analysis.
"""

import numpy as np

from PynPoint.Core.Processing import ProcessingModule


class SDIPreparationModule(ProcessingModule):
    """
    Module for preparing continuum for subtraction.
    """

    def __init__(self,
                 line_wvl,
                 cnt_wvl,
                 line_width,
                 cnt_width,
                 name_in="remove_frames",
                 image_in_tag="im_arr_cnt",
                 image_out_tag="im_arr_cnt2"):
        """
        Constructor of RemoveFramesModule.

        :param line_wvl: central wavelength of the line filter.
        :type line_wvl: float
        :param cnt_wvl: central wavelength of the continuum filter.
        :type cnt_wvl: float
        :param line_width: equivalent width of the line filter.
        :type line_width: float
        :param cnt_width: equivalent width of the continuum filter.
        :type cnt_width: float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(SDIPreparationModule, self).__init__(name_in)

        self.m_cnt_in_port = self.add_input_port(image_cnt_in_tag)
        
        self.m_stack_out_port = self.add_output_port(image_out_tag)

        self.m_line_wvl = line_wvl
        self.m_cnt_wvl = cnt_wvl
        self.m_line_width = line_width
        self.m_cnt_width = cnt_width

    def run(self):
        """
        Run method of the module. Normalizes for different filter widths, upscales the images to align PSF patterns and crops them to be of the same dimension as before.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        
        memory = self._m_config_port.get_attribute("MEMORY")
        cnt_size = self.m_cnt_in_port.get_shape()[1]
        
        def image_cutting(image_in, size):
            
            x_off = (image_in.shape[0] - size) / 2
            y_off = (image_in.shape[1] - size) / 2
                
            if size > image_in.shape[0] or size > image_in.shape[1]:
                raise ValueError("Input frame resolution smaller than target image resolution.")
                
            image_out = image_in[y_off:y_off+size, x_off:x_off+size]
            return image_out

        def prepare_cnt_images(image):
            
            width_factor = self.m_line_width/slef.m_cnt_width
            image *= width_factor
        
            wvl_factor = self.m_line_wvl/self.m_cnt_wvl
            sum_before = np.sum(tmp_cnt_im, axis=(1,2))
            tmp_cnt_im_rescaled = rescale(image=np.asarray(tmp_cnt_im, dtype=np.float64),
                                          scale = wvl_factor,
                                          order=5,
                                          mode="reflect")
            sum_after = np.sum(tmp_cnt_im_rescaled, axis=(1,2))
            tmp_cnt_im_rescaled *= (sum_before/sum_after)
        
            cnt_out=image_cutting(tmp_cnt_im_rescaled, cnt_size)
        
        return cnt_out
        
        
        self.apply_function_to_images(prepare_cnt_images,
                                      self.m_cnt_in_port,
                                      self.m_image_out_port,
                                      "Running SDI Preparation...",
                                      num_images_in_memory=memory)
        
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
            
        self.m_image_out_port.add_history_information("SDI", "continuum subtraction")
            
        self.m_image_in_port.close_database()
            
            
            
            


