import numpy as np

from PynPoint.core.Processing import ProcessingModule


class StackAndSubsetModule(ProcessingModule):

    def __init__(self,
                 name_in="stacking_subset",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr",
                 random_subset=None,
                 stacking=None):

        super(StackAndSubsetModule, self).__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_subset = random_subset
        self.m_stacking = stacking

    def run(self):

        if self.m_stacking in (None, False) and self.m_subset in (None, False):
            return

        tmp_data_shape = self.m_image_in_port.get_shape()

        # check if the random subset is available
        if tmp_data_shape[0] < self.m_subset:
            raise ValueError("The number of images of the destination subset is bigger than the "
                             "number of images in the source.")

        # get attributes
        tmp_files = self.m_image_in_port.get_attribute("Used_Files")
        number_of_images_per_file = self.m_image_in_port.get_attribute("NAXIS3")
        tmp_num_files = self.m_image_in_port.get_attribute("Num_Files")
        para_angles = self.m_image_in_port.get_attribute("NEW_PARA")

        # Do random subset first like in the old PynPoint
        if self.m_subset not in (None, False):
            number_of_elements = tmp_data_shape[0]  # number of images

            tmp_choice = np.random.choice(number_of_elements,
                                          self.m_subset,
                                          replace=False)

            tmp_choice = np.sort(tmp_choice)

            tmp_data = self.m_image_in_port[tmp_choice, :, :]
            para_angles = para_angles[tmp_choice]
            tmp_data_shape = tmp_data.shape

            if tmp_files is not None:
                if number_of_images_per_file is None and \
                                tmp_files == number_of_elements:
                    tmp_files = tmp_files[tmp_choice]
                elif number_of_images_per_file is not None:
                    tmp_files = tmp_files[tmp_choice/number_of_images_per_file]

            tmp_num_files = self.m_subset

            subset = True
        else:
            subset = False

        if self.m_stacking not in (False, None):

            num_new = int(np.floor(float(tmp_data_shape[0])/float(self.m_stacking)))
            para_new = np.zeros(num_new)
            im_arr_new = np.zeros([num_new,
                                   tmp_data_shape[1],
                                   tmp_data_shape[2]])
            for i in range(0, num_new):
                para_new[i] = para_angles[i*self.m_stacking:(i+1)*self.m_stacking].mean()
                if subset:
                    im_arr_new[i, ] = tmp_data[i*self.m_stacking:(i+1)*self.m_stacking, ]\
                        .mean(axis=0)
                else:
                    im_arr_new[i, ] = self.m_image_in_port[i * self.m_stacking:(i + 1)
                                                             * self.m_stacking, ] \
                                                           .mean(axis=0)

            # Update for saving
            tmp_data = im_arr_new
            para_angles = para_new

        # Save results
        self.m_image_out_port.set_all(tmp_data,
                                      keep_attributes=True)

        # Save Attributes
        # copy old attributes
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_attribute("NEW_PARA",
                                            para_angles,
                                            static=False)

        self.m_image_out_port.add_attribute("Used_Files",
                                            tmp_files,
                                            static=False)

        self.m_image_out_port.add_attribute("Num_Files",
                                            tmp_num_files,
                                            static=True)

        if self.m_stacking is None:
            history_stacked = "Nothing Stacked and "
        else:
            history_stacked = "Stacked every " + str(self.m_stacking) + " and "

        if self.m_subset is None:
            history_subset = "no subset chosen"
        else:
            history_subset = "picked randomly " + str(self.m_subset) + " frames."

        history = history_stacked + history_subset
        self.m_image_out_port.add_history_information("Subset",
                                                      history)
