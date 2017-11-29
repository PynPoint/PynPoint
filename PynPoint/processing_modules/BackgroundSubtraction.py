"""
Modules with background subtraction routines.
"""

# import copy
import warnings
import sys

from PynPoint.core.Processing import ProcessingModule
from PynPoint.processing_modules.SimpleTools import LocateStarModule

import numpy as np
from photutils import aperture_photometry, CircularAperture
from scipy.sparse.linalg import svds
from scipy.optimize import curve_fit

from astropy.io import fits


class MeanBackgroundSubtractionModule(ProcessingModule):
    """
    Module for mean background subtraction, only applicable for dithered data.
    """

    def __init__(self,
                 star_pos_shift=None,
                 name_in="mean_background_subtraction",
                 image_in_tag="im_arr",
                 image_out_tag="bg_cleaned_arr"):
        """
        Constructor of MeanBackgroundSubtractionModule.

        :param star_pos_shift: Frame index offset for the background subtraction. Typically equal
                               to the number of frames per dither location. If set to *None*, the
                               (non-static) NAXIS3 values from the FITS headers will be used.
        :type star_pos_shift: int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str
        :return: None
        """

        super(MeanBackgroundSubtractionModule, self).__init__(name_in)

        # add Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_star_prs_shift = star_pos_shift

    def run(self):
        """
        Run method of the module. Mean background subtraction which uses either a constant index
        offset or the (non-static) NAXIS3 values for the headers. The mean background is calculated
        from the cubes before and after the science cube.

        :return: None
        """

        # Use NAXIS3 values if star_pos_shift is None
        if self.m_star_prs_shift is None:
            self.m_star_prs_shift = self.m_image_in_port.get_attribute("NAXIS3")

        number_of_frames = self.m_image_in_port.get_shape()[0]

        # Check size of the input, only needed when a manual star_pos_shift is provided
        if not isinstance(self.m_star_prs_shift, np.ndarray) and \
               number_of_frames < self.m_star_prs_shift*2.0:
            raise ValueError("The input stack is to small for mean background subtraction. At least"
                             "one star position shift is needed.")


        # First subtraction to set up the output port array
        if isinstance(self.m_star_prs_shift, np.ndarray):
            # Modulo is needed when the index offset exceeds the total number of frames
            tmp_res = self.m_image_in_port[0] - \
                      self.m_image_in_port[(0 + self.m_star_prs_shift[0]) % number_of_frames]

        else:
            tmp_res = self.m_image_in_port[0] - \
                      self.m_image_in_port[(0 + self.m_star_prs_shift) % number_of_frames]

        # first subtraction is used to set up the output port array
        # calc mean
        if isinstance(self.m_star_prs_shift, np.ndarray):
            num_stacks = np.size(self.m_star_prs_shift)
        else:
            num_stacks = int(np.floor(number_of_frames/self.m_star_prs_shift))

        print "Subtracting background from stack-part " + str(1) + " of " + \
              str(num_stacks) + " stack-parts"

        if isinstance(self.m_star_prs_shift, np.ndarray):
            tmp_data = self.m_image_in_port[self.m_star_prs_shift[0]: \
                                            self.m_star_prs_shift[0]+self.m_star_prs_shift[1], \
                                            :, :]
            tmp_mean = np.mean(tmp_data, axis=0)

        else:
            tmp_data = self.m_image_in_port[self.m_star_prs_shift: self.m_star_prs_shift*2, :, :]
            tmp_mean = np.mean(tmp_data, axis=0)

        # init result port data
        tmp_res = self.m_image_in_port[0, :, :] - tmp_mean

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise NotImplementedError("Same input and output port not implemented yet.")
        else:
            self.m_image_out_port.set_all(tmp_res, data_dim=3)

        # clean first stack
        if isinstance(self.m_star_prs_shift, np.ndarray):
            tmp_data = self.m_image_in_port[1:self.m_star_prs_shift[0], :, :]

        else:
            tmp_data = self.m_image_in_port[1:self.m_star_prs_shift, :, :]

        tmp_data = tmp_data - tmp_mean
        self.m_image_out_port.append(tmp_data)  # TODO This will not work for same in and out port

        # process the rest of the stack
        if isinstance(self.m_star_prs_shift, np.ndarray):
            for i in range(1, num_stacks-1):
                print "Subtracting background from stack-part " + str(i+1) + " of " + \
                      str(num_stacks) + " stack-parts"
                # calc the mean (next)
                frame_ref = np.sum(self.m_star_prs_shift[0:i])
                tmp_data = self.m_image_in_port[frame_ref+self.m_star_prs_shift[i]: \
                                                frame_ref+self.m_star_prs_shift[i]+ \
                                                self.m_star_prs_shift[i+1], :, :]
                tmp_mean = np.mean(tmp_data, axis=0)
                # calc the mean (previous)
                tmp_data = self.m_image_in_port[frame_ref-self.m_star_prs_shift[i-1]: \
                                                frame_ref, :, :]
                tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

                # subtract mean
                tmp_data = self.m_image_in_port[frame_ref: frame_ref+self.m_star_prs_shift[i], \
                                                :, :]
                tmp_data = tmp_data - tmp_mean
                self.m_image_out_port.append(tmp_data)

            # mean subtraction of the last stack
            print "Subtracting background from stack-part " + str(num_stacks) + " of " + \
                  str(num_stacks) + " stack-parts"
            frame_ref = np.sum(self.m_star_prs_shift[0:num_stacks-1])
            tmp_data = self.m_image_in_port[frame_ref-self.m_star_prs_shift[num_stacks-2]:
                                            frame_ref, :, :]
            tmp_mean = np.mean(tmp_data, axis=0)
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)

        else:
            # the last and the one before will be performed afterwards
            top = int(np.ceil(number_of_frames /
                              self.m_star_prs_shift)) - 2

            for i in range(1, top, 1):
                print "Subtracting background from stack-part " + str(i+1) + " of " + \
                      str(num_stacks) + " stack-parts"
                # calc the mean (next)
                tmp_data = self.m_image_in_port[(i+1) * self.m_star_prs_shift:
                                                (i+2) * self.m_star_prs_shift,
                                                :, :]
                tmp_mean = np.mean(tmp_data, axis=0)
                # calc the mean (previous)
                tmp_data = self.m_image_in_port[(i-1) * self.m_star_prs_shift:
                                                (i+0) * self.m_star_prs_shift, :, :]
                tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

                # subtract mean
                tmp_data = self.m_image_in_port[(i+0) * self.m_star_prs_shift:
                                                (i+1) * self.m_star_prs_shift, :, :]
                tmp_data = tmp_data - tmp_mean
                self.m_image_out_port.append(tmp_data)

            # last and the one before
            # 1. ------------------------------- one before -------------------
            # calc the mean (previous)
            print "Subtracting background from stack-part " + str(top+1) + " of " + \
                  str(num_stacks) + " stack-parts"
            tmp_data = self.m_image_in_port[(top - 1) * self.m_star_prs_shift:
                                            (top + 0) * self.m_star_prs_shift, :, :]
            tmp_mean = np.mean(tmp_data, axis=0)
            # calc the mean (next)
            # "number_of_frames" is important if the last step is to huge
            tmp_data = self.m_image_in_port[(top + 1) * self.m_star_prs_shift:
                                            number_of_frames, :, :]

            tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

            # subtract mean
            tmp_data = self.m_image_in_port[top * self.m_star_prs_shift:
                                            (top + 1) * self.m_star_prs_shift, :, :]
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)

            # 2. ------------------------------- last -------------------
            # calc the mean (previous)
            print "Subtracting background from stack-part " + str(top+2) + " of " + \
                  str(num_stacks) + " stack-parts"
            tmp_data = self.m_image_in_port[(top + 0) * self.m_star_prs_shift:
                                            (top + 1) * self.m_star_prs_shift, :, :]
            tmp_mean = np.mean(tmp_data, axis=0)

            # subtract mean
            tmp_data = self.m_image_in_port[(top + 1) * self.m_star_prs_shift:
                                            number_of_frames, :, :]
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)
            # -----------------------------------------------------------

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_history_information("Background",
                                                      "mean subtraction")

        self.m_image_out_port.close_port()


class SimpleBackgroundSubtractionModule(ProcessingModule):
    """
    Module for simple background subtraction, only applicable for dithered data.
    """

    def __init__(self,
                 star_pos_shift=None,
                 name_in="background_subtraction",
                 image_in_tag="im_arr",
                 image_out_tag="bg_cleaned_arr"):
        """
        Constructor of SimpleBackgroundSubtractionModule.

        :param star_pos_shift: Frame index offset for the background subtraction. Typically equal
                               to the number of frames per dither location. If set to *None*, the
                               (non-static) NAXIS3 values from the FITS headers will be used.
        :type star_pos_shift: int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str
        :return: None
        """

        super(SimpleBackgroundSubtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_star_prs_shift = star_pos_shift

    def run(self):
        """
        Run method of the module. Simple background subtraction which uses either a constant index
        offset or the (non-static) NAXIS3 values for the headers.

        :return: None
        """

        # Use NAXIS3 values if star_pos_shift is None
        if self.m_star_prs_shift is None:
            self.m_star_prs_shift = self.m_image_in_port.get_attribute("NAXIS3")

        number_of_frames = self.m_image_in_port.get_shape()[0]

        # First subtraction to set up the output port array
        if isinstance(self.m_star_prs_shift, np.ndarray):
            # Modulo is needed when the index offset exceeds the total number of frames
            tmp_res = self.m_image_in_port[0] - \
                      self.m_image_in_port[(0 + self.m_star_prs_shift[0]) % number_of_frames]

        else:
            tmp_res = self.m_image_in_port[0] - \
                      self.m_image_in_port[(0 + self.m_star_prs_shift) % number_of_frames]

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            self.m_image_out_port[0] = tmp_res

        else:
            self.m_image_out_port.set_all(tmp_res, data_dim=3)

        # Background subtraction of the rest of the data
        if isinstance(self.m_star_prs_shift, np.ndarray):
            frame_count = 1
            for i, naxis_three in enumerate(self.m_star_prs_shift):
                for j in range(naxis_three):
                    if i == 0 and j == 0:
                        continue

                    else:
                        # TODO This will cause problems if the NAXIS3 value decreases and the
                        # amount of dithering positions is small, e.g. two dithering positions
                        # with subsequent NAXIS3 values of 20, 10, and 10. Also, the modulo does
                        # not guarentee to give a correct background frame.
                        if j == 0 and i < np.size(self.m_star_prs_shift)-1 and \
                                  self.m_star_prs_shift[i+1] > naxis_three:
                            warnings.warn("A small number (e.g., 2) of dither positions may give "
                                          "incorrect results when NAXIS3 is changing.")

                        tmp_res = self.m_image_in_port[frame_count] - \
                                  self.m_image_in_port[(frame_count + naxis_three) \
                                  % number_of_frames]

                    frame_count += 1

                    if self.m_image_in_port.tag == self.m_image_out_port.tag:
                        self.m_image_out_port[i] = tmp_res

                    else:
                        self.m_image_out_port.append(tmp_res)

        else:
            for i in range(1, number_of_frames):
                tmp_res = self.m_image_in_port[i] - \
                          self.m_image_in_port[(i + self.m_star_prs_shift) % number_of_frames]

                if self.m_image_in_port.tag == self.m_image_out_port.tag:
                    self.m_image_out_port[i] = tmp_res

                else:
                    self.m_image_out_port.append(tmp_res)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_history_information("Background",
                                                      "simple subtraction")

        self.m_image_out_port.close_port()


class PCABackgroundPreparationModule(ProcessingModule):
    """
    Module for preparing the PCA background subtraction.
    """

    def __init__(self,
                 select,
                 name_in="separate_star",
                 image_in_tag="im_arr",
                 star_out_tag="im_arr_star",
                 background_out_tag="im_arr_background"):
        """
        Constructor of PCABackgroundPreparationModule.

        select_star = dither positions, cubes_per_position, first_star_cube
        select_aperture = position, radius, threshold

        :param select: Tuple with the method ("dither" or "aperture") and parameters for separating
                       the star and background frames. The first is specified as ("dither",
                       dither_positions, cubes_per_position, first_star_cube), with
                       *dither_positions* the number of unique dither locations on the detector,
                       *cubes_per_position* the number of consecutive cubes per dither position, and
                       *first_star_cube* the index value of the first cube which contains the star
                       (Python indexing starts at zero). The second is specified as ("aperture,
                       (position_x, position_y), radius, threshold), with the position a two
                       element tuple of the aperture center, radius (pix) the aperture radius, and
                       threshold the fractional photometry threshold that is used to select the
                       star frames.
        :type select: tuple
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param star_out_tag: Tag of the database entry with frames that include the star. Should be
                             different from *image_in_tag*.
        :type star_out_tag: str
        :param background_out_tag: Tag of the the database entry with frames that contain only
                                   background and no star. Should be different from *image_in_tag*.
        :type background_out_tag: str
        :return: None
        """

        super(PCABackgroundPreparationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_star_out_port = self.add_output_port(star_out_tag)
        self.m_background_out_port = self.add_output_port(background_out_tag)
        
        self.m_select = select

        if self.m_select[0] is not "dither" and self.m_select[0] is not "aperture":
           raise ValueError("The first element of the 'select' tuple should be either 'dither' or 'aperture'.")

        elif self.m_select[0] is "dither":
            self.m_dither_positions = self.m_select[1]
            self.m_cubes_per_position = self.m_select[2]
            self.m_first_star_cube = self.m_select[3]

        elif self.m_select[0] is "aperture":
            
            self.m_position = self.m_select[1]
            self.m_radius = self.m_select[2]
            self.m_threshold = self.m_select[3]

        self.m_star_out_tag = star_out_tag

    def run(self):
        """
        Run method of the module. Performs aperture photometry on the median of each stack,
        separates the star and background frames by selecting photometric values larger than
        threshold times the maximum photometry, subtract the mean background from both the
        star and background frames, write the star and background data separately, and locate
        the star in each frame (required for the masking in the PCA background module).

        :return: None
        """

        if "NEW_PARA" not in self.m_image_in_port.get_all_non_static_attributes():
            raise ValueError("NEW_PARA not found in header. Parallactic angles should be "
                             "provided for all frames before PCA background subtraction.")

        parang = self.m_image_in_port.get_attribute("NEW_PARA")
        naxis_three = self.m_image_in_port.get_attribute("NAXIS3")

        if self.m_select[0] is "aperture" and self.m_position is None:
            self.m_position = (int(self.m_image_in_port.get_shape()[2]), \
                               int(self.m_image_in_port.get_shape()[1]))

        cube_mean = np.zeros((naxis_three.shape[0], self.m_image_in_port.get_shape()[2], \
                             self.m_image_in_port.get_shape()[1]))

        if self.m_select[0] is "dither":

            bg_frames = np.ones(naxis_three.shape[0], dtype=bool)

            # Mean of each cube
            count = 0
            for i, item in enumerate(naxis_three):
                cube_mean[i,] = np.median(self.m_image_in_port[count:count+item,], axis=0)
                count += item

            # Flag star and background cubes
            for i in range(self.m_first_star_cube, np.size(naxis_three), \
                           self.m_cubes_per_position*self.m_dither_positions):
                bg_frames[i:i+self.m_cubes_per_position] = False

            bg_indices = np.nonzero(bg_frames)[0]

        elif self.m_select[0] is "aperture":

            bg_frames = np.zeros(naxis_three.shape[0], dtype=bool)

            phot = np.zeros(naxis_three.shape[0])

            aperture = CircularAperture((self.m_position[0], self.m_position[1]), self.m_radius)

            # Aperture photometry on the mean of each cube
            count = 0
            for i, item in enumerate(naxis_three):
                cube_mean[i,] = np.median(self.m_image_in_port[count:count+item,], axis=0)
                phot_table = aperture_photometry(cube_mean[i,], aperture, method='exact')
                phot[i] = phot_table['aperture_sum'][0]
                count += item

            # Find star and background cubes
            for i, _ in enumerate(phot):
                if phot[i] < self.m_threshold*np.amax(phot):
                    bg_frames[i] = True

            bg_indices = np.nonzero(bg_frames)[0]

            if np.size(bg_indices) == 0:
                raise ValueError("No background cubes found. Try increasing the threshold and/or "
                                 "changing the aperture radius.")

            elif np.size(bg_indices) == phot.shape[0]:
                raise ValueError("No star cubes found. Try decreasing the threshold and/or changing "
                                 "the aperture radius.")

        bg_mean = np.zeros((self.m_image_in_port.get_shape()[2], self.m_image_in_port.get_shape()[1]))
        count = 0
        mean_count = 0
        for i, item in enumerate(naxis_three):
            if bg_frames[i]:
                bg_mean += np.mean(self.m_image_in_port[count:count+item,], axis=0)
                mean_count += 1
            count += item

        bg_mean /= float(mean_count)
        
        # hdunew = fits.HDUList()
        # hdunew.append(fits.ImageHDU(bg_mean))
        # hdunew.writeto('image.fits', overwrite=True)

        star_init = False
        background_init = False

        star_parang = np.empty(0)
        star_naxis_three = np.empty(0)

        background_parang = np.empty(0)
        background_naxis_three = np.empty(0)

        # Separate star and background cubes, and subtract mean background
        count = 0
        for i, item in enumerate(naxis_three):
            print "processed image "+str(count+1)+" of "+str(self.m_image_in_port.get_shape()[0])+" images"

            im_tmp = self.m_image_in_port[count:count+item,]

            # Background frames
            if bg_frames[i]:
                # Mean background of the cube
                # background = cube_mean[i]
                background = bg_mean

                # Subtract mean background, save data, and select corresponding NEW_PARA and NAXIS3
                if background_init:
                    self.m_background_out_port.append(im_tmp-background)

                    background_parang = np.append(background_parang, parang[count:count+item])
                    background_naxis_three = np.append(background_naxis_three, naxis_three[i])

                else:
                    self.m_background_out_port.set_all(im_tmp-background)

                    background_parang = parang[count:count+item]
                    background_naxis_three = naxis_three[i]

                    background_init = True

            # Star frames
            else:
                
                # Previous background cube
                if np.size(bg_indices[bg_indices < i]) > 0:
                    index_prev = np.amax(bg_indices[bg_indices < i])
                    bg_prev = cube_mean[index_prev,]

                # Next background cube
                if np.size(bg_indices[bg_indices > i]) > 0:
                    index_next = np.amin(bg_indices[bg_indices > i])
                    bg_next = cube_mean[index_next,]

                # Select background: previous, next, or mean of previous and next
                if i == 0:
                    background = bg_next

                elif i == np.size(naxis_three):
                    background = bg_prev
                    
                else:
                    background = (bg_prev+bg_next)/2.

                # Subtract mean background, save data, and select corresponding NEW_PARA and NAXIS3
                if star_init:
                    self.m_star_out_port.append(im_tmp-background)

                    star_parang = np.append(star_parang, parang[count:count+item])
                    star_naxis_three = np.append(star_naxis_three, naxis_three[i])

                else:
                    self.m_star_out_port.set_all(im_tmp-background)

                    star_parang = parang[count:count+item]
                    star_naxis_three = naxis_three[i]

                    star_init = True

            count += item

        # Star - Update attribute

        self.m_star_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_star_out_port.add_attribute("NEW_PARA", star_parang, static=False)
        self.m_star_out_port.add_attribute("NAXIS3", star_naxis_three, static=False)

        self.m_star_out_port.add_history_information("Star frames separated",
                                                     str(len(star_parang))+"/"+ \
                                                     str(len(parang))+" cubes")

        # Background - Update attributes

        self.m_background_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_background_out_port.add_attribute("NEW_PARA", background_parang, static=False)
        self.m_background_out_port.add_attribute("NAXIS3", background_naxis_three, static=False)

        self.m_background_out_port.add_history_information("Background frames separated",
                                                           str(len(background_parang))+"/"+ \
                                                           str(len(parang))+" cubes")

        # Close ports

        self.m_star_out_port.close_port()
        self.m_background_out_port.close_port()

        # Locate the position of the star

        locate_star = LocateStarModule(data_tag=self.m_star_out_tag,
                                       gaussian_fwhm=7)

        locate_star.connect_database(self._m_data_base)
        locate_star.run()


class PCABackgroundSubtractionModule(ProcessingModule):
    """
    Module for PCA background subtraction.
    """

    def __init__(self,
                 pca_number=20,
                 mask_radius=25,
                 mask_position="mean",
                 name_in="pca_background",
                 star_in_tag="im_star",
                 background_in_tag="im_background",
                 subtracted_out_tag="background_subtracted",
                 residuals_out_tag="background_residuals"):
        """
        Constructor of PCABackgroundSubtractionModule.

        :param pca_number: Number of principle components.
        :type pca_number: int
        :param mask_radius: Radius of the mask (pix).
        :type mask_radius: float
        :param mask_position: Position of the mask uses a single value ("mean") for all frames
                              or an value ("exact") for each frame separately.
        :type mask_position: str
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param star_in_tag: Tag of the input database entry with star frames.
        :type star_in_tag: str
        :param background_in_tag: Tag of the input database entry with the background frames.
        :type background_in_tag: str
        :param subtracted_out_tag: Tag of the output database entry with the background
                                   subtracted star frames.
        :type subtracted_out_tag: str
        :param residuals_out_tag: Tag of the output database entry with the residuals of the
                                  background subtraction.
        :type residuals_out_tag: str
        :return: None
        """

        super(PCABackgroundSubtractionModule, self).__init__(name_in)

        self.m_star_in_port = self.add_input_port(star_in_tag)
        self.m_background_in_port = self.add_input_port(background_in_tag)
        self.m_subtracted_out_port = self.add_output_port(subtracted_out_tag)
        self.m_residuals_out_port = self.add_output_port(residuals_out_tag)

        self.m_pca_number = pca_number
        self.m_mask_radius = mask_radius
        self.m_mask_position = mask_position

    def create_mask(self, mask_radius=25):
        """
        Method for creating a circular mask at the star position.
        """

        x = np.arange(0, self.im_star.shape[1], 1)
        y = np.arange(0, self.im_star.shape[2], 1)

        xx, yy = np.meshgrid(x, y)

        if self.m_mask_position == "mean":

            mask = np.ones(self.im_star[0,].shape)

            cent = [int(np.mean(self.star_position, axis=0)[0]), \
                    int(np.mean(self.star_position, axis=0)[1])]

            rr = np.sqrt((xx - cent[0])**2 + (yy - cent[1])**2)

            mask[rr < mask_radius] = 0.

        elif self.m_mask_position == "exact":

            mask = np.ones(self.im_star.shape)

            cent = [self.star_position[0], self.star_position[1]]
            cent = np.asarray(cent)

            for i in range(mask.shape[0]):
                rr = np.sqrt((xx - cent[0, i])**2 + (yy - cent[1, i])**2)
                mask[i, ][rr < mask_radius] = 0.
            
        self.mask = mask

    def mask_star(self):
        """
        Method for masking the central region around the star.
        """

        naxis_three = self.m_star_in_port.get_attribute("NAXIS3")

        # im_temp_star = copy.deepcopy(self.im_star)
        im_temp_star = np.copy(self.im_star)

        if self.m_mask_position == "mean":
            im_temp_star = im_temp_star[:,]*self.mask

        elif self.m_mask_position == "exact":
            count = 0
            for i, item in enumerate(naxis_three):
                im_temp_star[count:count+item,] = im_temp_star[count:count+item,]*self.mask[i,]
                count += item

        return im_temp_star

    def create_basis(self, im_arr):
        """
        Method for creating a set of principle components for a stack of images.
        """

        _, _, V = svds(im_arr.reshape(im_arr.shape[0],
                                      im_arr.shape[1]*im_arr.shape[2]),
                       k=self.m_pca_number)

        # V = V[::-1,]

        pca_basis = V.reshape(V.shape[0],
                              im_arr.shape[1],
                              im_arr.shape[2])

        return pca_basis

    def model_background(self, im_arr, basis):
        """
        Method for creating a model of the background.
        """

        def dot_product(x, *p):
            return np.dot(p, x)

        basis_reshaped = basis.reshape(basis.shape[0], -1)
        basis_reshaped_masked = (basis*self.mask).reshape(basis.shape[0], -1)

        fit_im_chi = np.zeros(im_arr.shape)
        fit_coeff_chi = np.zeros((im_arr.shape[0], basis.shape[0]))

        for i in xrange(im_arr.shape[0]):
            data_to_fit = im_arr[i,]

            init = np.ones(basis_reshaped_masked.shape[0])

            fitted = curve_fit(dot_product,
                               basis_reshaped_masked,
                               data_to_fit.reshape(-1),
                               init)

            fit_im = np.dot(fitted[0], basis_reshaped)
            fit_im = fit_im.reshape(data_to_fit.shape[0], data_to_fit.shape[1])

            fit_im_chi[i,] = fit_im
            # fit_coeff_chi[i,] = fitted[0]

        return fit_im_chi

    def run(self):
        """
        Run method of the module.

        :return: None
        """

        self.star_position = [self.m_star_in_port.get_attribute("STAR_POSITION_X"),
                              self.m_star_in_port.get_attribute("STAR_POSITION_Y")]

        self.im_star = self.m_star_in_port.get_all()
        self.im_background = self.m_background_in_port.get_all()

        # Mask the PSF
        self.create_mask(mask_radius=self.m_mask_radius)

        # Create basis set of principal components from the background frames
        basis_pca = self.create_basis(self.im_background)

        # Fit the star frames with self.m_pca_number principal components
        fit_im = self.model_background(self.mask_star(), basis_pca)

        # PCA background subtracted

        self.m_subtracted_out_port.set_all(self.im_star[:,] - fit_im[:,])

        self.m_subtracted_out_port.copy_attributes_from_input_port(self.m_star_in_port)

        self.m_subtracted_out_port.add_history_information("Background",
                                                           "PCA subtraction")

        self.m_subtracted_out_port.close_port()

        # PCA background residuals

        hdunew = fits.HDUList()
        hdunew.append(fits.ImageHDU(fit_im))
        hdunew.writeto('image.fits', overwrite=True)

        self.m_residuals_out_port.set_all(fit_im)

        self.m_residuals_out_port.copy_attributes_from_input_port(self.m_star_in_port)

        self.m_residuals_out_port.add_history_information("Background",
                                                          "PCA residuals")

        self.m_residuals_out_port.close_port()
