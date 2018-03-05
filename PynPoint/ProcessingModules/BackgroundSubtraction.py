"""
Modules with background subtraction routines.
"""

import sys

import numpy as np

from scipy.sparse.linalg import svds
from scipy.optimize import curve_fit

from PynPoint.Util.Progress import progress
from PynPoint.Core.Processing import ProcessingModule
from PynPoint.ProcessingModules.ImageResizing import CropImagesModule, CombineTagsModule
from PynPoint.ProcessingModules.StarAlignment import StarExtractionModule


class SimpleBackgroundSubtractionModule(ProcessingModule):
    """
    Module for simple background subtraction. Only applicable on data obtained with dithering.
    """

    def __init__(self,
                 shift,
                 name_in="simple_background",
                 image_in_tag="im_arr",
                 image_out_tag="bg_sub_arr"):
        """
        Constructor of SimpleBackgroundSubtractionModule.

        :param shift: Frame index offset for the background subtraction. Typically equal to the
                      number of frames per dither location.
        :type shift: int
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

        self.m_shift = shift

    def run(self):
        """
        Run method of the module. Simple background subtraction with a constant index offset.

        :return: None
        """

        nframes = self.m_image_in_port.get_shape()[0]

        subtract = self.m_image_in_port[0] - self.m_image_in_port[(0 + self.m_shift) % nframes]

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            self.m_image_out_port[0] = subtract
        else:
            self.m_image_out_port.set_all(subtract, data_dim=3)

        for i in range(1, nframes):
            progress(i, nframes, "Running SimpleBackgroundSubtractionModule...")

            subtract = self.m_image_in_port[i] - self.m_image_in_port[(i + self.m_shift) % nframes]

            if self.m_image_in_port.tag == self.m_image_out_port.tag:
                self.m_image_out_port[i] = subtract
            else:
                self.m_image_out_port.append(subtract)

        sys.stdout.write("Running SimpleBackgroundSubtractionModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("Background", "simple subtraction")
        self.m_image_out_port.close_database()


class MeanBackgroundSubtractionModule(ProcessingModule):
    """
    Module for mean background subtraction. Only applicable on data obtained with dithering.
    """

    def __init__(self,
                 shift=None,
                 cubes=1,
                 name_in="mean_background",
                 image_in_tag="im_arr",
                 image_out_tag="bg_sub_arr"):
        """
        Constructor of MeanBackgroundSubtractionModule.

        :param shift: Image index offset for the background subtraction. Typically equal to the
                      number of frames per dither location. If set to *None*, the NFRAMES attribute
                      will be used to select the background frames automatically. The *cubes*
                      argument should also be set with *shift=None*.
        :type shift: int
        :param cubes: Number of consecutive cubes per dithering position.
        :type cubes: int
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

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_shift = shift
        self.m_cubes = cubes

    def run(self):
        """
        Run method of the module. Mean background subtraction which uses either a constant index
        offset or the NFRAMES attributes. The mean background is calculated from the cubes before
        and after the science cube.

        :return: None
        """

        # Use NFRAMES values if shift=None
        if self.m_shift is None:
            self.m_shift = self.m_image_in_port.get_attribute("NFRAMES")

        nframes = self.m_image_in_port.get_shape()[0]

        if not isinstance(self.m_shift, np.ndarray) and nframes < self.m_shift*2.0:
            raise ValueError("The input stack is too small for a mean background subtraction. The "
                             "position of the star should shift at least once.")

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("The tag of the input port should be different from the output port.")

        # Number of substacks
        if isinstance(self.m_shift, np.ndarray):
            nstacks = np.size(self.m_shift)
        else:
            nstacks = int(np.floor(nframes/self.m_shift))

        # First mean subtraction to set up the output port array
        if isinstance(self.m_shift, np.ndarray):
            next_start = np.sum(self.m_shift[0:self.m_cubes])
            next_end = np.sum(self.m_shift[0:2*self.m_cubes])

            if 2*self.m_cubes > np.size(self.m_shift):
                raise ValueError("Not enough frames available for the background subtraction.")

            bg_data = self.m_image_in_port[next_start:next_end, ]
            bg_mean = np.mean(bg_data, axis=0)

        else:
            bg_data = self.m_image_in_port[self.m_shift:2*self.m_shift, ]
            bg_mean = np.mean(bg_data, axis=0)

        # Initiate the result port data with the first frame
        bg_sub = self.m_image_in_port[0, ] - bg_mean
        self.m_image_out_port.set_all(bg_sub, data_dim=3)

        # Mean subtraction of the first stack (minus the first frame)
        if isinstance(self.m_shift, np.ndarray):
            tmp_data = self.m_image_in_port[1:next_start, ]
            tmp_data = tmp_data - bg_mean
            self.m_image_out_port.append(tmp_data)

        else:
            tmp_data = self.m_image_in_port[1:self.m_shift, ]
            tmp_data = tmp_data - bg_mean
            self.m_image_out_port.append(tmp_data)

        # Processing of the rest of the data
        if isinstance(self.m_shift, np.ndarray):
            for i in range(self.m_cubes, nstacks, self.m_cubes):
                progress(i, nstacks, "Running MeanBackgroundSubtractionModule...")

                prev_start = np.sum(self.m_shift[0:i-self.m_cubes])
                prev_end = np.sum(self.m_shift[0:i])

                next_start = np.sum(self.m_shift[0:i+self.m_cubes])
                next_end = np.sum(self.m_shift[0:i+2*self.m_cubes])

                # calc the mean (previous)
                tmp_data = self.m_image_in_port[prev_start:prev_end, ]
                tmp_mean = np.mean(tmp_data, axis=0)

                if i < nstacks-self.m_cubes:
                    # calc the mean (next)
                    tmp_data = self.m_image_in_port[next_start:next_end, ]
                    tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

                # subtract mean
                tmp_data = self.m_image_in_port[prev_end:next_start, ]
                tmp_data = tmp_data - tmp_mean
                self.m_image_out_port.append(tmp_data)

        else:
            # the last and the one before will be performed afterwards
            top = int(np.ceil(nframes/self.m_shift)) - 2

            for i in range(1, top, 1):
                progress(i, top, "Running MeanBackgroundSubtractionModule...")

                # calc the mean (next)
                tmp_data = self.m_image_in_port[(i+1)*self.m_shift:(i+2)*self.m_shift, ]
                tmp_mean = np.mean(tmp_data, axis=0)

                # calc the mean (previous)
                tmp_data = self.m_image_in_port[(i-1)*self.m_shift:(i+0)*self.m_shift, ]
                tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

                # subtract mean
                tmp_data = self.m_image_in_port[(i+0)*self.m_shift:(i+1)*self.m_shift, ]
                tmp_data = tmp_data - tmp_mean
                self.m_image_out_port.append(tmp_data)

            # last and the one before
            # 1. ------------------------------- one before -------------------
            # calc the mean (previous)
            tmp_data = self.m_image_in_port[(top-1)*self.m_shift:(top+0)*self.m_shift, ]
            tmp_mean = np.mean(tmp_data, axis=0)

            # calc the mean (next)
            # "nframes" is important if the last step is to huge
            tmp_data = self.m_image_in_port[(top+1)*self.m_shift:nframes, ]
            tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

            # subtract mean
            tmp_data = self.m_image_in_port[top*self.m_shift:(top+1)*self.m_shift, ]
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)

            # 2. ------------------------------- last -------------------
            # calc the mean (previous)
            tmp_data = self.m_image_in_port[(top+0)*self.m_shift:(top+1)*self.m_shift, ]
            tmp_mean = np.mean(tmp_data, axis=0)

            # subtract mean
            tmp_data = self.m_image_in_port[(top+1)*self.m_shift:nframes, ]
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)
            # -----------------------------------------------------------

        sys.stdout.write("Running MeanBackgroundSubtractionModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("Background", "mean subtraction")
        self.m_image_out_port.close_database()


class PCABackgroundPreparationModule(ProcessingModule):
    """
    Module for preparing the PCA background subtraction.
    """

    def __init__(self,
                 dither,
                 mean=False,
                 name_in="separate_star",
                 image_in_tag="im_arr",
                 star_out_tag="im_arr_star",
                 background_out_tag="im_arr_background"):
        """
        Constructor of PCABackgroundPreparationModule.

        :param dither: Tuple with the parameters for separating the star and background frames.
                       The tuple should contain three values (positions, cubes, first) with
                       *dither* the number of unique dithering position, *cubes* the number of
                       consecutive cubes per dithering position, and *first* the index value of the
                       first cube which contains the star (Python indexing starts at zero). Sorting
                       is based on the DITHER_X and DITHER_Y attributes  when *cubes_per_position*
                       is set to None.
        :type select: tuple
        :param mean: Subtract the mean pixel value from each image separately, both star and
                     background frames.
        :type mean: bool
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

        if len(dither) != 3:
            raise ValueError("The 'dither' tuple should contain three integer values.")

        self.m_dither = dither
        self.m_mean = mean

    def run(self):
        """
        Run method of the module. Separates the star and background frames, subtracts the mean
        background from both the star and background frames, and writes the star and background
        frames separately.

        :return: None
        """

        self.m_star_out_port.del_all_data()
        self.m_star_out_port.del_all_attributes()

        self.m_background_out_port.del_all_data()
        self.m_background_out_port.del_all_attributes()

        nframes = self.m_image_in_port.get_attribute("NFRAMES")

        if "NEW_PARA" in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute("NEW_PARA")
        else:
            parang = None

        # Mean of each cube
        cube_mean = np.zeros((nframes.shape[0], self.m_image_in_port.get_shape()[2], \
                             self.m_image_in_port.get_shape()[1]))

        count = 0
        for i, item in enumerate(nframes):
            cube_mean[i, ] = np.mean(self.m_image_in_port[count:count+item, ], axis=0)
            count += item

        # Flag star and background cubes
        if self.m_dither[1] is None:
            dither_x = self.m_image_in_port.get_attribute("DITHER_X")
            dither_y = self.m_image_in_port.get_attribute("DITHER_Y")

            star = np.logical_and(dither_x == self.m_dither[2][0], dither_y == self.m_dither[2][1])
            bg_frames = np.invert(star)

        else:
            bg_frames = np.ones(nframes.shape[0], dtype=bool)
            for i in range(self.m_dither[2]*self.m_dither[1], np.size(nframes), \
                           self.m_dither[1]*self.m_dither[0]):
                bg_frames[i:i+self.m_dither[1]] = False

        bg_indices = np.nonzero(bg_frames)[0]

        background_nframes = np.empty(0, dtype=np.int64)
        star_nframes = np.empty(0, dtype=np.int64)

        if parang is not None:
            background_parang = np.empty(0, dtype=np.float64)
            star_parang = np.empty(0, dtype=np.float64)

        # Separate star and background cubes. Subtract mean background.
        count = 0
        for i, item in enumerate(nframes):
            progress(i, len(nframes), "Running PCABackgroundPreparationModule...")

            im_tmp = self.m_image_in_port[count:count+item, ]

            if self.m_mean:
                for j in enumerate(item):
                    im_tmp[j, ] -= np.mean(im_tmp[j, ])

            # Background frames
            if bg_frames[i]:
                # Mean background of the cube
                background = cube_mean[i, ]

                # Subtract mean background, save data, and select corresponding NEW_PARA and NFRAMES
                self.m_background_out_port.append(im_tmp-background)
                background_nframes = np.append(background_nframes, nframes[i])
                if parang is not None:
                    background_parang = np.append(background_parang, parang[count:count+item])

            # Star frames
            else:
                # Previous background cube
                if np.size(bg_indices[bg_indices < i]) > 0:
                    index_prev = np.amax(bg_indices[bg_indices < i])
                    bg_prev = cube_mean[index_prev, ]

                else:
                    bg_prev = None

                # Next background cube
                if np.size(bg_indices[bg_indices > i]) > 0:
                    index_next = np.amin(bg_indices[bg_indices > i])
                    bg_next = cube_mean[index_next, ]

                else:
                    bg_next = None

                # Select background: previous, next, or mean of previous and next
                if bg_prev is None and bg_next is not None:
                    background = bg_next

                elif bg_prev is not None and bg_next is None:
                    background = bg_prev

                elif bg_prev is not None and bg_next is not None:
                    background = (bg_prev+bg_next)/2.

                else:
                    raise ValueError("Neither previous nor next background frames found.")

                # Subtract mean background, save data, and select corresponding NEW_PARA and NFRAMES
                self.m_star_out_port.append(im_tmp-background)
                star_nframes = np.append(star_nframes, nframes[i])
                if parang is not None:
                    star_parang = np.append(star_parang, parang[count:count+item])

            count += item

        sys.stdout.write("Running PCABackgroundPreparationModule... [DONE]\n")
        sys.stdout.flush()

        self.m_star_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_star_out_port.add_attribute("NFRAMES", star_nframes, static=False)

        if parang is not None:
            self.m_star_out_port.add_attribute("NEW_PARA", star_parang, static=False)

        self.m_star_out_port.add_history_information("Star frames separated",
                                                     str(sum(star_nframes))+"/"+ \
                                                     str(sum(nframes)))

        self.m_background_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_background_out_port.add_attribute("NFRAMES", background_nframes, static=False)

        if parang is not None:
            self.m_background_out_port.add_attribute("NEW_PARA", background_parang, static=False)

        self.m_background_out_port.add_history_information("Background frames separated",
                                                           str(len(background_nframes))+"/"+ \
                                                           str(len(nframes)))

        self.m_star_out_port.close_database()


class PCABackgroundSubtractionModule(ProcessingModule):
    """
    Module for PCA background subtraction.
    """

    def __init__(self,
                 pca_number=60,
                 mask_radius=0.7,
                 name_in="pca_background",
                 star_in_tag="im_star",
                 background_in_tag="im_background",
                 subtracted_out_tag="background_subtracted",
                 residuals_out_tag=None):
        """
        Constructor of PCABackgroundSubtractionModule.

        :param pca_number: Number of principle components.
        :type pca_number: int
        :param mask_radius: Radius of the mask (arcsec).
        :type mask_radius: float
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
        if residuals_out_tag is not None:
            self.m_residuals_out_port = self.add_output_port(residuals_out_tag)

        self.m_pca_number = pca_number
        self.m_mask_radius = mask_radius
        self.m_residuals_out_tag = residuals_out_tag

    def _create_mask(self, mask_radius, star_position, num_frames):
        """
        Method for creating a circular mask at the star position.
        """

        im_dim = self.m_star_in_port[0, ].shape

        x_grid = np.arange(0, im_dim[0], 1)
        y_grid = np.arange(0, im_dim[1], 1)

        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

        mask = np.ones((num_frames, im_dim[0], im_dim[1]))

        cent_x = star_position[:, 0]
        cent_y = star_position[:, 1]

        for i in range(num_frames):
            rr_grid = np.sqrt((xx_grid - cent_x[i])**2 + (yy_grid - cent_y[i])**2)
            mask[i, ][rr_grid < mask_radius] = 0.

        return mask

    def _create_basis(self, im_arr):
        """
        Method for creating a set of principle components for a stack of images.
        """

        _, _, v_svd = svds(im_arr.reshape(im_arr.shape[0],
                                          im_arr.shape[1]*im_arr.shape[2]),
                           k=self.m_pca_number)

        v_svd = v_svd[::-1, ]

        pca_basis = v_svd.reshape(v_svd.shape[0], im_arr.shape[1], im_arr.shape[2])

        return pca_basis

    def _model_background(self, basis, im_arr, mask):
        """
        Method for creating a model of the background.
        """

        def _dot_product(x_dot, *p):
            return np.dot(p, x_dot)

        fit_im_chi = np.zeros(im_arr.shape)
        # fit_coeff_chi = np.zeros((im_arr.shape[0], basis.shape[0]))

        basis_reshaped = basis.reshape(basis.shape[0], -1)

        for i in xrange(im_arr.shape[0]):
            basis_reshaped_masked = (basis*mask[i]).reshape(basis.shape[0], -1)

            data_to_fit = im_arr[i, ]

            init = np.ones(basis_reshaped_masked.shape[0])

            fitted = curve_fit(_dot_product,
                               basis_reshaped_masked,
                               data_to_fit.reshape(-1),
                               init)

            fit_im = np.dot(fitted[0], basis_reshaped)
            fit_im = fit_im.reshape(data_to_fit.shape[0], data_to_fit.shape[1])

            fit_im_chi[i, ] = fit_im
            # fit_coeff_chi[i, ] = fitted[0]

        return fit_im_chi

    def run(self):
        """
        Run method of the module. Creates a PCA basis set of the background frames, masks the PSF
        in the star frames, fits the star frames with a linear combination of the principle
        components, and writes the background subtracted star frames and the background residuals
        that are subtracted.

        :return: None
        """

        self.m_subtracted_out_port.del_all_data()
        self.m_subtracted_out_port.del_all_attributes()

        if self.m_residuals_out_tag is not None:
            self.m_residuals_out_port.del_all_data()
            self.m_residuals_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_star_in_port.get_attribute("PIXSCALE")
        star_position = self.m_star_in_port.get_attribute("STAR_POSITION")

        self.m_mask_radius /= pixscale

        im_background = self.m_background_in_port.get_all()

        sys.stdout.write("Creating PCA basis set...")
        sys.stdout.flush()
        basis_pca = self._create_basis(im_background)
        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        n_image = self.m_star_in_port.get_shape()[0]
        if memory == -1 or memory >= n_image:
            n_stack = 1
        else:
            n_stack = int(float(n_image)/float(memory))

        for i in range(n_stack):
            progress(i, n_stack, "Calculating background model...")

            if memory == -1 or memory >= n_image:
                frame_start = 0
                frame_end = n_image
                im_star = self.m_star_in_port.get_all()

            else:
                frame_start = i*memory
                frame_end = i*memory+memory
                im_star = self.m_star_in_port[frame_start:frame_end, ]

            mask = self._create_mask(self.m_mask_radius,
                                     star_position[frame_start:frame_end, :],
                                     frame_end-frame_start)

            im_star_mask = im_star*mask
            fit_im = self._model_background(basis_pca, im_star_mask, mask)

            self.m_subtracted_out_port.append(im_star-fit_im)
            if self.m_residuals_out_tag is not None:
                self.m_residuals_out_port.append(fit_im)

        sys.stdout.write("Calculating background model... [DONE]\n")
        sys.stdout.flush()

        if memory <= n_image and n_image%memory > 0:
            frame_start = n_stack*memory
            frame_end = n_image

            im_star = self.m_star_in_port[frame_start:frame_end, ]

            mask = self._create_mask(self.m_mask_radius,
                                     star_position[frame_start:frame_end, :],
                                     frame_end-frame_start)

            im_star_mask = im_star*mask
            fit_im = self._model_background(basis_pca, im_star_mask, mask)

            self.m_subtracted_out_port.append(im_star-fit_im)
            if self.m_residuals_out_tag is not None:
                self.m_residuals_out_port.append(fit_im)

        self.m_subtracted_out_port.copy_attributes_from_input_port(self.m_star_in_port)
        self.m_subtracted_out_port.add_history_information("Background",
                                                           "PCA subtraction")

        if self.m_residuals_out_tag is not None:
            self.m_residuals_out_port.copy_attributes_from_input_port(self.m_star_in_port)
            self.m_residuals_out_port.add_history_information("Background",
                                                              "PCA residuals")

        self.m_subtracted_out_port.close_database()


class DitheringBackgroundModule(ProcessingModule):
    """
    Module for PCA-based background subtraction of data with dithering. This is a wrapper that
    applies the processing modules required for the PCA background subtraction.
    """

    def __init__(self,
                 name_in="background_dithering",
                 image_in_tag="im_arr",
                 image_out_tag="im_bg",
                 center=None,
                 cubes=None,
                 shape=(100, 100),
                 gaussian=0.15,
                 subframe=None,
                 pca_number=60,
                 mask_radius=0.7,
                 **kwargs):
        """
        Constructor of DitheringBackgroundModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str
        :param center: Tuple with the centers of the dithering positions, e.g. ((x0,y0), (x1,y1)).
                       The order of the coordinates should correspond to the order in which the
                       star is present. If *center* and *cubes* are both set to None then sorting
                       and subtracting of the background frames is based on DITHER_X and DITHER_Y.
                       If *center* is specified and *cubes* is set to None then the DITHER_X and
                       DITHER_Y attributes will be used for sorting and subtracting of the
                       background but not for selecting the dithering positions.
        :type center: tuple, int
        :param cubes: Number of consecutive cubes per dither position. If *cubes* is set to None
                      then sorting and subtracting of the background frames is based on DITHER_X
                      and DITHER_Y.
        :type cubes: int
        :param shape: Tuple (delta_x, delta_y) with the image size that is cropped at the specified
                      dither positions.
        :type shape: tuple, int
        :param gaussian: Full width at half maximum (arcsec) of the Gaussian kernel that is used
                         to smooth the image before the star is located.
        :type gaussian: float
        :param subframe: Size (pix) of the subframe that is used to search for the star. Cropping
                         of the subframe is done around the center of the dithering position.
        :type subframe: float
        :param pca_number: Number of principle components.
        :type pca_number: int
        :param mask_radius: Radius of the mask that is placed at the location of the star (arcsec).
        :type mask_radius: float
        :param \**kwargs:
            See below.

        :Keyword arguments:
             **crop** (*bool*) -- Skip the step of selecting and cropping of the dithering
                                  positions if set to False.
             **prepare** (*bool*) -- Skip the step of preparing the PCA background subtraction if
                                     set to False.
             **pca_background** (*bool*) -- Skip the step of the PCA background subtraction if set
                                            to False.
             **combine** (*str*) -- Combine the mean background subtracted ("mean") or PCA
                                    background subtracted ("pca") frames. This step is ignored if
                                    set to None.

        :return: None
        """

        if "crop" in kwargs:
            self.m_crop = kwargs["crop"]
        else:
            self.m_crop = True

        if "prepare" in kwargs:
            self.m_prepare = kwargs["prepare"]
        else:
            self.m_prepare = True

        if "pca_background" in kwargs:
            self.m_pca_background = kwargs["pca_background"]
        else:
            self.m_pca_background = True

        if "combine" in kwargs:
            self.m_combine = kwargs["combine"]
        else:
            self.m_combine = True

        super(DitheringBackgroundModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        self.m_center = center
        self.m_cubes = cubes
        self.m_shape = shape
        self.m_gaussian = gaussian
        self.m_subframe = subframe
        self.m_pca_number = pca_number
        self.m_mask_radius = mask_radius

        self.m_image_in_tag = image_in_tag
        self.m_image_out_tag = image_out_tag

    def run(self):
        """
        Run method of the module. Cuts out the detector sections at the different dither positions,
        prepares the PCA background subtraction, applies a bad pixel correction, locates the star
        in each image, runs the PCA background subtraction, combines the output from the different
        dither positions is written to a single database tag.

        :return: None
        """

        if self.m_cubes is None:
            dither_x = self.m_image_in_port.get_attribute("DITHER_X")
            dither_y = self.m_image_in_port.get_attribute("DITHER_Y")

            dither_xy = np.zeros((dither_x.shape[0], 2))
            dither_xy[:, 0] = dither_x
            dither_xy[:, 1] = dither_y

            _, index = np.unique(dither_xy, axis=0, return_index=True)
            dither = dither_xy[np.sort(index)]

            npix = self.m_image_in_port.get_shape()[1]

            if self.m_center is None:
                self.m_center = np.copy(dither)
                self.m_center += float(npix)/2.
                self.m_center = tuple(map(tuple, self.m_center))

            else:
                if np.size(dither, axis=0) != np.size(self.m_center, axis=0):
                    raise ValueError("Number of specified center positions should be equal to the "
                                     "number of unique dithering positions.")

        n_dither = np.size(self.m_center, 0)

        if self.m_cubes is None:
            star_pos = np.copy(dither)
        else:
            star_pos = np.arange(0, n_dither, 1)

        tags = []

        for i, position in enumerate(self.m_center):
            if self.m_crop or self.m_prepare or self.m_pca_background:
                print "Processing dither position "+str(i+1)+" out of "+str(n_dither)+"..."
                print "Center position =", position

                if self.m_cubes is None and self.m_center is not None:
                    print "DITHER_X, DITHER_Y =", tuple(star_pos[i])

            if self.m_crop:
                crop = CropImagesModule(shape=self.m_shape,
                                        center=position,
                                        name_in="crop"+str(i),
                                        image_in_tag=self.m_image_in_tag,
                                        image_out_tag="dither"+str(i+1))

                crop.connect_database(self._m_data_base)
                crop.run()

            if self.m_prepare:
                prepare = PCABackgroundPreparationModule(dither=(n_dither,
                                                                 self.m_cubes,
                                                                 star_pos[i]),
                                                         mean=False,
                                                         name_in="prepare"+str(i),
                                                         image_in_tag="dither"+str(i+1),
                                                         star_out_tag="star"+str(i+1),
                                                         background_out_tag="background"+str(i+1))

                prepare.connect_database(self._m_data_base)
                prepare.run()

            if self.m_pca_background:
                if self.m_subframe is None:
                    position = None
                else:
                    position = (None, None, self.m_subframe)

                star = StarExtractionModule(name_in="star"+str(i),
                                            image_in_tag="star"+str(i+1),
                                            image_out_tag=None,
                                            image_size=None,
                                            fwhm_star=self.m_gaussian,
                                            position=position)

                star.connect_database(self._m_data_base)
                star.run()

                pca = PCABackgroundSubtractionModule(pca_number=self.m_pca_number,
                                                     mask_radius=self.m_mask_radius,
                                                     name_in="pca_background"+str(i),
                                                     star_in_tag="star"+str(i+1),
                                                     background_in_tag="background"+str(i+1),
                                                     subtracted_out_tag="pca_sub"+str(i+1),
                                                     residuals_out_tag=None)

                pca.connect_database(self._m_data_base)
                pca.run()

            if self.m_combine == "mean":
                tags.append("star"+str(i+1))

            elif self.m_combine == "pca":
                tags.append("pca_sub"+str(i+1))

            if self.m_crop or self.m_prepare or self.m_pca_background:
                print "Processing dither position "+str(i+1)+" out of "+str(n_dither)+"... [DONE]"

        if self.m_combine is not None:
            combine = CombineTagsModule(name_in="combine",
                                        check_attr=True,
                                        image_in_tags=tags,
                                        image_out_tag=self.m_image_out_tag)

            combine.connect_database(self._m_data_base)
            combine.run()


class NoddingBackgroundModule(ProcessingModule):
    """
    Module for background subtraction of data obtained with nodding (e.g., NACO AGPM data).
    """

    def __init__(self,
                 name_in="sky_subtraction",
                 sky_in_tag="sky_arr",
                 science_in_tag="im_arr",
                 image_out_tag="im_arr",
                 mode="both"):
        """
        Constructor of NoddingBackgroundModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param sky_in_tag: Tag of the database entry with sky frames that are read as input.
        :type sky_in_tag: str
        :param science_data_in_tag: Tag of the database entry with science frames that are read as
                                    input.
        :type science_data_in_tag: str
        :param image_out_tag: Tag of the database entry with sky subtracted images that are written
                              as output.
        :type image_out_tag: str

        :return: None
        """

        super(NoddingBackgroundModule, self).__init__(name_in=name_in)

        self.m_sky_in_port = self.add_input_port(sky_in_tag)
        self.m_science_in_port = self.add_input_port(science_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_time_stamps = []

        if mode in ["next", "previous", "both"]:
            self.m_mode = mode
        else:
            raise ValueError("Mode needs to be next, previous or both.")

    def _create_time_stamp_list(self):
        """
        Internal method for assigning a time stamp, based on the exposure number ID, to each cube
        of sky and science frames.
        """

        class TimeStamp:
            def __init__(self,
                         time,
                         sky_or_science,
                         index):
                self.m_time = time
                self.m_sky_or_science = sky_or_science
                self.m_index = index

            def __repr__(self):
                return repr((self.m_time,
                             self.m_sky_or_science,
                             self.m_index))

        exp_no = self.m_sky_in_port.get_attribute("EXP_NO")

        for i, item in enumerate(exp_no):
            self.m_time_stamps.append(TimeStamp(item,
                                                "SKY",
                                                i))

        exp_no = self.m_science_in_port.get_attribute("EXP_NO")
        nframes = self.m_science_in_port.get_attribute("NFRAMES")

        current = 0
        for i, item in enumerate(exp_no):
            self.m_time_stamps.append(TimeStamp(item,
                                                "SCIENCE",
                                                slice(current, current+nframes[i])))
            current += nframes[i]

        self.m_time_stamps = sorted(self.m_time_stamps, key=lambda time_stamp: time_stamp.m_time)

    def calc_sky_frame(self,
                       index_of_science_data):
        """
        Method for finding the required sky frame (next, previous, or the mean of next and
        previous) by comparing the time stamp of the science frame with preceding and following
        sky frames.
        """

        # check if there is at least one SKY in the database
        if not any(x.m_sky_or_science == "SKY" for x in self.m_time_stamps):
            raise ValueError('List of time stamps does not contain any SKY frames')

        def search_for_next_sky():
            for i in range(index_of_science_data, len(self.m_time_stamps)):
                if self.m_time_stamps[i].m_sky_or_science == "SKY":
                    return self.m_sky_in_port[self.m_time_stamps[i].m_index, ]

            # no next sky found look for previous sky
            return search_for_previous_sky()

        def search_for_previous_sky():
            for i in reversed(range(0, index_of_science_data)):
                if self.m_time_stamps[i].m_sky_or_science == "SKY":
                    return self.m_sky_in_port[self.m_time_stamps[i].m_index, ]

            # no previous sky found look for next sky
            return search_for_next_sky()

        if self.m_mode == "next":
            return search_for_next_sky()

        if self.m_mode == "previous":
            return search_for_previous_sky()

        if self.m_mode == "both":
            previous_sky = search_for_previous_sky()
            next_sky = search_for_next_sky()
            return (previous_sky + next_sky)/2.0

    def run(self):
        """
        Run method of the module. Create list of time stamps, get sky and science frames, and
        subtract the sky background from the science frames.

        :return: None
        """

        self._create_time_stamp_list()

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        for i, time_entry in enumerate(self.m_time_stamps):
            progress(i, len(self.m_time_stamps), "Running NoddingBackgroundModule...")

            if time_entry.m_sky_or_science == "SKY":
                continue

            sky = self.calc_sky_frame(i)
            science = self.m_science_in_port[time_entry.m_index, ]

            self.m_image_out_port.append(science - sky[None, ], data_dim=3)

        sys.stdout.write("Running NoddingBackgroundModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_science_in_port)

        self.m_image_out_port.add_history_information("Background", "nodding")

        self.m_image_out_port.close_database()
