import numpy as np


from PynPoint.Processing import ProcessingModule


class StackAndSubsetModule(ProcessingModule):

    def __init__(self,
                 name_in="stacking_subset",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr",
                 random_subset=None,
                 stacking=None):

        super(StackAndSubsetModule, self).__init__(name_in)

        # Ports
        self.m_image_in_tag = image_in_tag
        self.add_input_port(image_in_tag)

        self.m_image_out_tag = image_out_tag
        self.add_output_port(image_out_tag)

        self.m_subset = random_subset
        self.m_stacking = stacking

    def run(self):

        # TODO raise Error if ransub is bigger than the number of images

        if self.m_stacking in (None, False) and self.m_subset in (None, False):
            return

        # get the data
        tmp_data = self._m_input_ports[self.m_image_in_tag].get_all()

        # get attributes
        tmp_files = self._m_input_ports[self.m_image_in_tag].get_attribute("Used_Files")
        tmp_num_files = self._m_input_ports[self.m_image_in_tag].get_attribute("Num_Files")

        # Do random subset first like in the old PynPoint
        if self.m_subset not in (None, False):
            number_of_elements = tmp_data.shape[0] # number of images

            tmp_choice = np.random.choice(number_of_elements,
                                          self.m_subset,
                                          replace=False)

            tmp_data = tmp_data[tmp_choice, :, :]

            if tmp_files is not None:
                tmp_files = tmp_files[tmp_choice]

            tmp_num_files = self.m_subset

        self._m_output_ports[self.m_image_out_tag].set_all(tmp_data,
                                                           keep_attributes=True)
        self._m_output_ports[self.m_image_out_tag].add_attribute("Used_Files",
                                                                 tmp_files,
                                                                 static=False)
        self._m_output_ports[self.m_image_out_tag].add_attribute("Num_Files",
                                                                 tmp_num_files,
                                                                 static=True)

        '''
        if not self.m_stacking in (None, False):'''
        # TODO stacking






