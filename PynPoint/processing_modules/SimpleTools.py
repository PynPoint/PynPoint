"""
Modules with simple pre-processing tools.
"""

from PynPoint.core import ProcessingModule
from skimage.transform import rescale
from scipy.ndimage import shift
import numpy as np
from scipy import ndimage


class CutAroundCenterModule(ProcessingModule):
    """
    Module for cropping around the center of an image.
    """

    def __init__(self,
                 new_shape,
                 name_in="cut_around_center",
                 image_in_tag="im_arr",
                 image_out_tag="cut_im_arr",
                 number_of_images_in_memory=100):
        """
        Constructor of CutAroundCenterModule.

        :param new_shape: Tuple (delta_x, delta_y) with the new image size.
        :type new_shape: tuple, int
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

        super(CutAroundCenterModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_shape = new_shape

    def run(self):
        """
        Run method of the module. Reduces the image size by cropping around the center of the
        original image.

        :return: None
        """

        def image_cutting(image_in,
                          shape_in):

            shape_of_input = image_in.shape

            if shape_in[0] > shape_of_input[0] or shape_in[1] > shape_of_input[1]:
                raise ValueError("Input frame resolution smaller than target image resolution.")

            x_off = (shape_of_input[0] - shape_in[0]) / 2
            y_off = (shape_of_input[1] - shape_in[1]) / 2

            return image_in[y_off: shape_in[1] + y_off, x_off: shape_in[0] + x_off]

        self.apply_function_to_images(image_cutting,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(self.m_shape,),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Cropped image size to",
                                                      str(self.m_shape))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class CutAroundPositionModule(ProcessingModule):
    """
    Module for cropping around a given position of an image.
    """

    def __init__(self,
                 new_shape,
                 center_of_cut,
                 name_in="cut_around_position",
                 image_in_tag="im_arr",
                 image_out_tag="cut_im_arr",
                 number_of_images_in_memory=100):
        """
        Constructor of CutAroundPositionModule.

        :param new_shape: Tuple (delta_x, delta_y) with the new image size.
        :type new_shape: tuple, int
        :param center_of_cut: Tuple (x0, y0) with the new image center. Python indexing starts at 0.
        :type center_of_cut: tuple, int
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

        super(CutAroundPositionModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_shape = new_shape
        self.m_center_of_cut = center_of_cut

    def run(self):
        """
        Run method of the module. Reduces the image size by cropping around an given position.

        :return: None
        """

        def image_cutting(image_in,
                          shape_in,
                          center_of_cut_in):

            shape_of_input = image_in.shape

            if shape_in[0] > shape_of_input[0] or shape_in[1] > shape_of_input[1]:
                raise ValueError("Input frame resolution smaller than target image resolution.")

            x_off = center_of_cut_in[0] - (shape_in[0] / 2)
            y_off = center_of_cut_in[1] - (shape_in[1] / 2)

            return image_in[y_off: shape_in[1] + y_off, x_off: shape_in[0] + x_off]

        self.apply_function_to_images(image_cutting,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(self.m_shape, self.m_center_of_cut),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Cropped image size to",
                                                      str(self.m_shape))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class ScaleFramesModule(ProcessingModule):
    """
    Module for rescaling of an image.
    """

    def __init__(self,
                 scaling_factor,
                 name_in="scaling",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_scaled",
                 number_of_images_in_memory=100):
        """
        Constructor of ScaleFramesModule.

        :param scaling_factor: Scaling factor for upsampling (*scaling_factor* > 1) and downsampling
                               (0 < *scaling_factor* < 1).
        :type scaling_factor: float
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

        super(ScaleFramesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_scaling = scaling_factor

    def run(self):
        """
        Run method of the module. Rescales an image with a fifth order spline interpolation and a
        reflecting boundary condition.

        :return: None
        """

        def image_scaling(image_in,
                          scaling):

            sum_before = np.sum(image_in)
            tmp_image = rescale(image=np.asarray(image_in,
                                                 dtype=np.float64),
                                scale=(scaling,
                                       scaling),
                                order=5,
                                mode="reflect")

            sum_after = np.sum(tmp_image)
            return tmp_image * (sum_before / sum_after)

        self.apply_function_to_images(image_scaling,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(self.m_scaling,),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Scaled by a factor of",
                                                      str(self.m_scaling))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class ShiftForCenteringModule(ProcessingModule):
    """
    Module for shifting of an image.
    """

    def __init__(self,
                 shift_vector,
                 name_in="shift",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_shifted",
                 number_of_images_in_memory=100):
        """
        Constructor of ShiftForCenteringModule.

        :param shift_vector: Tuple (delta_y, delta_x) with the shift in both directions.
        :type new_shape: tuple, float
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

        super(ShiftForCenteringModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_shift_vector = shift_vector

    def run(self):
        """
        Run method of the module. Shifts an image with a fifth order spline interpolation.

        :return: None
        """

        def image_shift(image_in):

            return shift(image_in, self.m_shift_vector, order=5)

        self.apply_function_to_images(image_shift,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Shifted by",
                                                      str(self.m_shift_vector))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class RemoveFrameModule(ProcessingModule):
    """
    Module for removing a single frame.
    """

    def __init__(self,
                 frame_number,
                 name_in="remove_frame",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_remove",
                 num_image_in_memory=100):
        """
        Constructor of RemoveFrameModule.

        :param frame_number: Frame number to be removed. Python indexing starts at 0.
        :type frame_number: int
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

        super(RemoveFrameModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_frame_number = frame_number
        self.m_image_memory = num_image_in_memory

    def run(self):
        """
        Run method of the module. Removes single frame and saves the data and attributes.

        :return: None
        """

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        num_subsets = int(self.m_image_in_port.get_shape()[0]/self.m_image_memory)

        for i in range(num_subsets):

            tmp_im = self.m_image_in_port[i*self.m_image_memory:(i+1)*self.m_image_memory, :, :]

            if self.m_frame_number > i*self.m_image_memory and \
                                   self.m_frame_number < (i+1)*self.m_image_memory:
                tmp_im = np.delete(tmp_im,
                                   self.m_frame_number%self.m_image_memory,
                                   axis=0)

            if i == 0:
                self.m_image_out_port.set_all(tmp_im, keep_attributes=True)
            else:
                self.m_image_out_port.append(tmp_im)
            
        tmp_im = self.m_image_in_port[num_subsets*self.m_image_memory: \
                                      self.m_image_in_port.get_shape()[0], :, :]
        
        self.m_image_out_port.append(tmp_im)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_history_information("Removed frame number",
                                                      str(self.m_frame_number))

        self.m_image_out_port.close_port()


class CombineResArrsModule(ProcessingModule):
    '''
    Module to combine residual arrays produced by the PSFSubtractionModule after rotating by their position angles.
    This module is useful for pseudoADI data, in which the star is observed at two distinct position angles,
    as opposed to using pupil tracking mode.
    '''

    def __init__(self,
                 res_arr_in_tag_list,
                 name_in="combine_res_arrs",
                 res_arr_rot_out_tag="res_rot",
                 res_mean_tag="res_mean",
                 res_median_tag="res_median",
                 res_var_tag="res_var",
                 res_rot_mean_clip_tag="res_rot_mean_clip",
                 num_image_in_memory=100):
        """
        Constructor of RemoveFrameModule.

        :param res_arr_in_tag_list: List of input database tags, each corresponding to a residual array produced by the PSFSubtractionModule.
        :type frame_number: list of str
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param res_arr_rot_out_tag: Tag of the database entry that is written as output of the rotated residual array.
        :type res_arr_rot_out_tag: str
        :param res_mean_tag: Tag of the database entry that is written as output of the mean of the rotated residual array.
        :type res_mean_tag: str
        :param res_median_tag: Tag of the database entry that is written as output of the median of the rotated residual array.
        :type res_median_tag: str
        :param res_var_tag: Tag of the database entry that is written as output of the variance of the rotated residual array.
        :type res_var_tag: str
        :param res_rot_mean_clip_tag: Tag of the database entry that is written as output of the clipped mean of the rotated residual array.
        :type res_rot_mean_clip_tag: str
        :param num_image_in_memory: Number of frames that are simultaneously loaded into the memory.
        :type num_image_in_memory: int
        :return: None
        """

        super(CombineResArrsModule, self).__init__(name_in)

        # Inputs
        self.m_res_arr_in_port_list = [self.add_input_port(res_arr_in_tag)  for res_arr_in_tag in res_arr_in_tag_list]

        # Outputs
        self.m_res_arr_rot_out_port = self.add_output_port(res_arr_rot_out_tag)
        self.m_res_mean_port = self.add_output_port(res_mean_tag)
        self.m_res_median_port = self.add_output_port(res_median_tag)
        self.m_res_var_port = self.add_output_port(res_var_tag)
        self.m_res_rot_mean_clip_port = self.add_output_port(res_rot_mean_clip_tag)

        self.m_image_memory = num_image_in_memory

    def run(self):
        """
        Run method of the module. Gets the position angles of each residual array, rotates the residuals,
        and finally combines the frames.

        :return: None
        """

        # get information for first res_arr
        shape_of_first = self.m_res_arr_in_port_list[0].get_shape()
        num_frames = [shape_of_first[0]]


        # loop over rest of input ports
        for r in range(1, len(self.m_res_arr_in_port_list)):
            #get number of frames
            tmp_shape = self.m_res_arr_in_port_list[r].get_shape()
            num_frames.append(tmp_shape[0])

            # check that all input ports have same shape in axis 1 and 2 as first port
            if tmp_shape[1:3] != shape_of_first[1:3]:
                raise ValueError('Input ports given in res_arr_in_tag_list do not have same shape in axes 1 and 2.')


        # rotate the input residual arrays
        res_rot = np.zeros((sum(num_frames),self.m_res_arr_in_port_list[0].get_shape()[1],
                            self.m_res_arr_in_port_list[0].get_shape()[2]))   # assumes all res_arrs have same shape
        for i, in_port in enumerate(self.m_res_arr_in_port_list):

            # get position angle of this residual array
            posang_attr = in_port.get_attribute('ESO ADA POSANG')
            posang_unique = np.unique(posang_attr)
            if len(posang_unique) == 1:  # if each residual array only has one position angle, as expected for pseudoADI data
                posang_rot = posang_unique[0]
            else:
                raise ValueError('Position angle varies in residual array in %s data tag: this is not supported by this module.'
                                 % (in_port.tag))

            # check of posang is static or not
            tmp_res_arr = in_port.get_all()

            tmp_res_rot_arr = ndimage.rotate(tmp_res_arr,
                                         posang_rot,
                                         axes=(2,1),
                                         reshape=False)

            res_rot[sum(num_frames[0:i]):sum(num_frames[0:i+1]),:,:] = tmp_res_rot_arr

        # combined rotated residual arrays
        # create mean
        tmp_res_rot_mean = np.mean(res_rot,
                                   axis=0)

        # create median
        tmp_res_rot_median = np.median(res_rot,
                                       axis=0)

        # create variance
        res_rot_temp = res_rot.copy()
        for i in range(0,
                       res_rot_temp.shape[0]):

            res_rot_temp[i, ] -= - tmp_res_rot_mean
        res_rot_var = (res_rot_temp**2.).sum(axis=0)
        tmp_res_rot_var = res_rot_var

        # create mean clip
        res_rot_mean_clip = np.zeros(self.m_res_arr_in_port_list[0].get_shape())

        for i in range(0, res_rot_mean_clip.shape[0]):
            for j in range(0, res_rot_mean_clip.shape[1]):
                temp = res_rot[:, i, j]
                if temp.var() > 0.0:
                    a = temp - temp.mean()
                    b1 = a.compress((a < 3.0*np.sqrt(a.var())).flat)
                    b2 = b1.compress((b1 > (-1.0)*3.0*np.sqrt(a.var())).flat)
                    res_rot_mean_clip[i, j] = temp.mean() + b2.mean()

        # save results
        self.m_res_arr_rot_out_port.set_all(res_rot)
        self.m_res_mean_port.set_all(tmp_res_rot_mean)
        self.m_res_median_port.set_all(tmp_res_rot_median)
        self.m_res_var_port.set_all(tmp_res_rot_var)
        self.m_res_rot_mean_clip_port.set_all(res_rot_mean_clip)
