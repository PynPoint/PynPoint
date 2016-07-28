"""
Pypline Modules for the detection and interpolation of bad pixels
"""

from PynPoint.core.Processing import ProcessingModule


class BadPixelCleaningSigmaFilterModule(ProcessingModule):

    def __init__(self,
                 name_in="sigma_filtering",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_bp_clean"):

        super(BadPixelCleaningSigmaFilterModule, self).__init__(name_in)

        # Ports
        self.m_image_in_tag = image_in_tag
        self.add_input_port(image_in_tag)

        self.m_image_out_tag = image_out_tag
        self.add_output_port(image_out_tag)

    def run(self):
        pass