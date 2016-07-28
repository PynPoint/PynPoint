import warnings

from PynPoint.core.Processing import ProcessingModule


class DarkSubtractionModule(ProcessingModule):

    def __init__(self,
                 name_in="dark_subtraction",
                 image_in_tag="im_arr",
                 dark_in_tag="dark_arr",
                 image_out_tag="dark_sub_arr",
                 number_of_images_in_memory=100):

        super(DarkSubtractionModule, self).__init__(name_in=name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_dark_in_port = self.add_input_port(dark_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory

    def run(self):

        def dark_subtraction_image(image_in,
                                   dark_in):
            return image_in - dark_in

        dark = self.m_dark_in_port.get_all()

        # check dim of the dark. We need only the first frame.
        if dark.ndim == 3:
            tmp_dark = dark[0]
        elif dark.ndim == 2:
            tmp_dark = dark
        else:
            raise ValueError("Dimension of input dark not supported. "
                             "Only 2D and 3D dark array can be used.")

        # check shape of the dark and cut if needed
        shape_of_input = (self.m_image_in_port.get_shape()[1],
                          self.m_image_in_port.get_shape()[2])

        if tmp_dark.shape[0] < shape_of_input[0] or tmp_dark.shape[1] < shape_of_input[1]:
            raise ValueError("Input dark resolution smaller than image resolution.")

        if not (tmp_dark.shape == shape_of_input):
            # cut the dark first
            dark_shape = tmp_dark.shape
            x_off = (dark_shape[0] - shape_of_input[0]) / 2
            y_off = (dark_shape[1] - shape_of_input[1]) / 2
            tmp_dark = tmp_dark[x_off: shape_of_input[0] + x_off,
                       y_off: shape_of_input[1] + y_off]

            warnings.warn("The given dark has a different shape than the input images and was cut "
                          "around the center in order to get the same resolution. If the position "
                          "of the dark does not match the input frame you have to cut the dark "
                          "before using this module.")

        self.apply_function_to_images(dark_subtraction_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(tmp_dark,),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        history = "simple subtraction"
        self.m_image_out_port.add_attribute("history: dark subtraction",
                                            history)

        self.m_image_out_port.close_port()


class FlatSubtractionModule(ProcessingModule):

    def __init__(self,
                 name_in="flat_subtraction",
                 image_in_tag="dark_sub_arr",
                 flat_in_tag="flat_arr",
                 image_out_tag="flat_sub_arr",
                 number_of_images_in_memory=100):

        super(FlatSubtractionModule, self).__init__(name_in=name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_flat_in_port = self.add_input_port(flat_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory

    def run(self):

        def flat_subtraction_image(image_in,
                                   flat_in):
            return image_in - flat_in

        flat = self.m_flat_in_port.get_all()

        # check dim of the dark. We need only the first frame.
        if flat.ndim == 3:
            tmp_flat = flat[0]
        elif flat.ndim == 2:
            tmp_flat = flat
        else:
            raise ValueError("Dimension of input dark not supported. "
                             "Only 2D and 3D dark array can be used.")

        # check shape of the dark and cut if needed
        shape_of_input = (self.m_image_in_port.get_shape()[1],
                          self.m_image_in_port.get_shape()[2])

        if tmp_flat.shape[0] < shape_of_input[0] or tmp_flat.shape[1] < shape_of_input[1]:
            raise ValueError("Input dark resolution smaller than image resolution.")

        if not (tmp_flat.shape == shape_of_input):
            # cut the dark first
            dark_shape = tmp_flat.shape
            x_off = (dark_shape[0] - shape_of_input[0]) / 2
            y_off = (dark_shape[1] - shape_of_input[1]) / 2
            x_off += 1 # TODO find solution for NACO
            tmp_flat = tmp_flat[x_off: shape_of_input[0] + x_off,
                       y_off: shape_of_input[1] + y_off]

            warnings.warn("The given dark has a different shape than the input images and was cut "
                          "around the center in order to get the same resolution. If the position "
                          "of the dark does not match the input frame you have to cut the dark "
                          "before using this module.")

        self.apply_function_to_images(flat_subtraction_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(tmp_flat,),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        history = "simple division"
        self.m_image_out_port.add_attribute("history: flat subtraction",
                                            history)

        self.m_image_out_port.close_port()