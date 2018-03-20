"""
Modules with tools for preparation to SDI analysis.
"""

import numpy as np
import sys

from PynPoint.Core.Processing import ProcessingModule
from PynPoint.ProcessingModules import CropImagesModule, ScaleImagesModule


class SDIPreparationModule(ProcessingModule):
    """
    Module for preparing continuum for subtraction.
    """

    def __init__(self,
                 line_wvl,
                 cnt_wvl,
                 line_width,
                 cnt_width,
                 name_in="SDI_preparation",
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

        self.m_image_in_port = self.add_input_port(image_in_tag)
        
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_line_wvl = line_wvl
        self.m_cnt_wvl = cnt_wvl
        self.m_line_width = line_width
        self.m_cnt_width = cnt_width
        self.m_image_in_tag = image_in_tag
        self.m_cnt_out_tag = image_out_tag

    def run(self):
        """
        Run method of the module. Normalizes for different filter widths, upscales the images to align PSF patterns and crops them to be of the same dimension as before.

        :return: None
        """
        
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        
        width_factor = self.m_line_width/self.m_cnt_width
        wvl_factor = self.m_line_wvl/self.m_cnt_wvl
        im_size = self.m_image_in_port.get_shape()[1]
        
        sys.stdout.write("Starting SDI preparation... \n")
        
        scaling = ScaleImagesModule(scaling_dim = wvl_factor,
                                      scaling_flux = width_factor,
                                      name_in="scaling",
                                      image_in_tag=self.m_image_in_tag,
                                      image_out_tag="im_arr_scaled")
                                      
        scaling.connect_database(self._m_data_base)
        scaling.run()
        
        
        crop = CropImagesModule(im_size*pixscale,
                                center=None,
                                name_in="crop",
                                image_in_tag="im_arr_scaled",
                                image_out_tag=self.m_cnt_out_tag)
        
        crop.connect_database(self._m_data_base)
        crop.run()
        sys.stdout.write("SDI preparation finished... \n")
            
        self.m_image_in_port.close_database()
            
            
            
            


