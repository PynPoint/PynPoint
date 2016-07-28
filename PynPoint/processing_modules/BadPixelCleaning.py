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
        self.m_image_in_port = self.add_input_port(image_in_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        def make_black(image_in):
            for i in range(image_in.shape[0]):
                for j in range(image_in.shape[1]):
                    image_in[i, j] = 0
            return image_in

        self.apply_function_to_images(make_black,
                                      self.m_image_in_port,
                                      self.m_image_out_port)