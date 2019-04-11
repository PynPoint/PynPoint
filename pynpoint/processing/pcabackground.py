"""
Pipeline modules for PCA-based background subtraction.
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import math

import numpy as np

from scipy.sparse.linalg import svds
from scipy.optimize import curve_fit
from six.moves import map
from six.moves import range

from pynpoint.core.processing import ProcessingModule
from pynpoint.processing.resizing import CropImagesModule
from pynpoint.processing.stacksubset import CombineTagsModule
from pynpoint.processing.psfpreparation import SortParangModule
from pynpoint.util.module import progress, memory_frames, locate_star


class PCABackgroundPreparationModule(ProcessingModule):
    """
    Module for preparing the PCA background subtraction.
    """

    def __init__(self,
                 dither,
                 name_in="separate_star",
                 image_in_tag="im_arr",
                 star_out_tag="im_arr_star",
                 mean_out_tag="im_arr_mean",
                 background_out_tag="im_arr_background"):
        """
        Constructor of PCABackgroundPreparationModule.

        Parameters
        ----------
        dither : tuple(int, int, int)
            Tuple with the parameters for separating the star and background frames. The tuple
            should contain three values (positions, cubes, first) with *positions* the number
            of unique dithering position, *cubes* the number of consecutive cubes per dithering
            position, and *first* the index value of the first cube which contains the star
            (Python indexing starts at zero). Sorting is based on the DITHER_X and DITHER_Y
            attributes when *cubes* is set to None.
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        star_out_tag : str
            Tag of the database entry with frames that include the star. Should be different
            from *image_in_tag*.
        mean_out_tag : str
            Tag of the database entry with frames that include the star and for which a mean
            background subtraction has been applied. Should be different from *image_in_tag*.
        background_out_tag : str
            Tag of the the database entry with frames that contain only background and no star.
            Should be different from *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(PCABackgroundPreparationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_star_out_port = self.add_output_port(star_out_tag)
        self.m_mean_out_port = self.add_output_port(mean_out_tag)
        self.m_background_out_port = self.add_output_port(background_out_tag)

        if len(dither) != 3:
            raise ValueError("The 'dither' tuple should contain three integer values.")

        self.m_dither = dither

    def _prepare(self):
        nframes = self.m_image_in_port.get_attribute("NFRAMES")

        cube_mean = np.zeros((nframes.shape[0],
                              self.m_image_in_port.get_shape()[2],
                              self.m_image_in_port.get_shape()[1]))

        count = 0
        for i, item in enumerate(nframes):
            cube_mean[i, ] = np.mean(self.m_image_in_port[count:count+item, ], axis=0)
            count += item

        if self.m_dither[1] is None:
            dither_x = self.m_image_in_port.get_attribute("DITHER_X")
            dither_y = self.m_image_in_port.get_attribute("DITHER_Y")

            star = np.logical_and(dither_x == self.m_dither[2][0],
                                  dither_y == self.m_dither[2][1])

            bg_frames = np.invert(star)

        else:
            bg_frames = np.ones(nframes.shape[0], dtype=bool)

            for i in range(self.m_dither[2]*self.m_dither[1],
                           np.size(nframes),
                           self.m_dither[1]*self.m_dither[0]):

                bg_frames[i:i+self.m_dither[1]] = False

        return bg_frames, cube_mean

    def _separate(self, bg_frames, bg_indices, parang, cube_mean):

        def _initialize():
            background_nframes = np.empty(0, dtype=np.int64)
            star_nframes = np.empty(0, dtype=np.int64)

            background_index = np.empty(0, dtype=np.int64)
            star_index = np.empty(0, dtype=np.int64)

            if parang is None:
                background_parang = None
                star_parang = None

            else:
                background_parang = np.empty(0, dtype=np.float64)
                star_parang = np.empty(0, dtype=np.float64)

            return star_index, star_parang, star_nframes, background_index, \
                   background_parang, background_nframes

        def _select_background(i):

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

            return background

        nframes = self.m_image_in_port.get_attribute("NFRAMES")
        index = self.m_image_in_port.get_attribute("INDEX")

        star_index, star_parang, star_nframes, background_index, background_parang, \
            background_nframes = _initialize()

        # Separate star and background cubes. Subtract mean background.
        count = 0
        for i, item in enumerate(nframes):
            progress(i, len(nframes), "Running PCABackgroundPreparationModule...")

            im_tmp = self.m_image_in_port[count:count+item, ]

            # Background frames
            if bg_frames[i]:
                background = cube_mean[i, ]
                self.m_background_out_port.append(im_tmp)

                background_nframes = np.append(background_nframes, nframes[i])
                background_index = np.append(background_index, index[count:count+item])

                if parang is not None:
                    background_parang = np.append(background_parang, parang[count:count+item])

            # Star frames
            else:
                background = _select_background(i)

                self.m_star_out_port.append(im_tmp)
                self.m_mean_out_port.append(im_tmp-background)

                star_nframes = np.append(star_nframes, nframes[i])
                star_index = np.append(star_index, index[count:count+item])

                if parang is not None:
                    star_parang = np.append(star_parang, parang[count:count+item])

            count += item

        return star_index, star_parang, star_nframes, background_index, \
               background_parang, background_nframes

    def run(self):
        """
        Run method of the module. Separates the star and background frames, subtracts the mean
        background from both the star and background frames, and writes the star and background
        frames separately.

        Returns
        -------
        NoneType
            None
        """

        self.m_star_out_port.del_all_data()
        self.m_star_out_port.del_all_attributes()

        self.m_mean_out_port.del_all_data()
        self.m_mean_out_port.del_all_attributes()

        self.m_background_out_port.del_all_data()
        self.m_background_out_port.del_all_attributes()

        if "PARANG" in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute("PARANG")
        else:
            parang = None

        bg_frames, cube_mean = self._prepare()

        bg_indices = np.nonzero(bg_frames)[0]

        star_index, star_parang, star_nframes, background_index, background_parang, \
            background_nframes = self._separate(bg_frames, bg_indices, parang, cube_mean)

        sys.stdout.write("Running PCABackgroundPreparationModule... [DONE]\n")
        sys.stdout.flush()

        history = "star = "+str(sum(star_nframes))+", background"+str(len(background_nframes))
        self.m_star_out_port.copy_attributes(self.m_image_in_port)
        self.m_star_out_port.add_history("PCABackgroundPreparationModule", history)
        self.m_star_out_port.add_attribute("NFRAMES", star_nframes, static=False)
        self.m_star_out_port.add_attribute("INDEX", star_index, static=False)

        if parang is not None:
            self.m_star_out_port.add_attribute("PARANG", star_parang, static=False)

        self.m_mean_out_port.copy_attributes(self.m_image_in_port)
        self.m_mean_out_port.add_history("PCABackgroundPreparationModule", history)
        self.m_mean_out_port.add_attribute("NFRAMES", star_nframes, static=False)
        self.m_mean_out_port.add_attribute("INDEX", star_index, static=False)

        if parang is not None:
            self.m_mean_out_port.add_attribute("PARANG", star_parang, static=False)

        self.m_background_out_port.copy_attributes(self.m_image_in_port)
        self.m_background_out_port.add_history("PCABackgroundPreparationModule", history)
        self.m_background_out_port.add_attribute("NFRAMES", background_nframes, static=False)
        self.m_background_out_port.add_attribute("INDEX", background_index, static=False)

        if parang is not None:
            self.m_background_out_port.add_attribute("PARANG", background_parang, static=False)

        self.m_star_out_port.close_port()


class PCABackgroundSubtractionModule(ProcessingModule):
    """
    Module for PCA based background subtraction. See Hunziker et al. 2018 for details.
    """

    def __init__(self,
                 pca_number=60,
                 mask_star=0.7,
                 mask_planet=None,
                 subtract_mean=False,
                 subframe=None,
                 gaussian=0.15,
                 name_in="pca_background",
                 star_in_tag="im_star",
                 background_in_tag="im_background",
                 residuals_out_tag="background_subtracted",
                 fit_out_tag=None,
                 mask_out_tag=None):
        """
        Constructor of PCABackgroundSubtractionModule.

        Parameters
        ----------
        pca_number : int
            Number of principal components.
        mask_star : float
            Radius of the central mask (arcsec).
        mask_planet : tuple(float, float, float, float)
            Separation (arcsec), position angle (deg) measured in counterclockwise direction with
            respect to upward direction, additional rotation angle (deg), and radius (arcsec) of
            the mask, (sep, angle, extra_rot, radius). No mask is used when set to None.
        subtract_mean : bool
            The mean of the background images is subtracted from both the star and background
            images before the PCA basis is constructed.
        gaussian : float
            Full width at half maximum (arcsec) of the Gaussian kernel that is used to smooth the
            image before the star is located.
        subframe : float
            Size (arcsec) of the subframe that is used to find the star. Cropping of the subframe
            is done around the center of the image. The full images is used if set to None.
        name_in : str
            Tag of the database entry with the star images.
        star_in_tag : str
            Tag of the database entry with the star images.
        background_in_tag : str
            Tag of the database entry with the background images.
        residuals_out_tag : str
            Tag of the database entry with the residuals of the star images after the background
            subtraction.
        fit_out_tag : str
            Tag of the database entry with the fitted background. No data is written when set to
            None.
        mask_out_tag : str
            Tag of the database entry with the mask. No data is written when set to None.

        Returns
        -------
        NoneType
            None
        """

        super(PCABackgroundSubtractionModule, self).__init__(name_in)

        self.m_star_in_port = self.add_input_port(star_in_tag)
        self.m_background_in_port = self.add_input_port(background_in_tag)
        self.m_residuals_out_port = self.add_output_port(residuals_out_tag)

        if fit_out_tag is None:
            self.m_fit_out_port = None
        else:
            self.m_fit_out_port = self.add_output_port(fit_out_tag)

        if mask_out_tag is None:
            self.m_mask_out_port = None
        else:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)

        self.m_pca_number = pca_number
        self.m_mask_star = mask_star
        self.m_mask_planet = mask_planet
        self.m_subtract_mean = subtract_mean
        self.m_gaussian = gaussian
        self.m_subframe = subframe

    def run(self):
        """
        Run method of the module. Creates a PCA basis set of the background frames, masks the PSF
        in the star frames and optionally an off-axis point source, fits the star frames with a
        linear combination of the principal components, and writes the residuals of the background
        subtracted images.

        Returns
        -------
        NoneType
            None
        """

        def _create_mask(radius, position, nimages):
            """
            Method for creating a circular mask at the star or planet position.
            """

            npix = self.m_star_in_port.get_shape()[1]

            x_grid = np.arange(0, npix, 1)
            y_grid = np.arange(0, npix, 1)

            xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

            mask = np.ones((nimages, npix, npix))

            cent_x = position[:, 1]
            cent_y = position[:, 0]

            for i in range(nimages):
                rr_grid = np.sqrt((xx_grid - cent_x[i])**2 + (yy_grid - cent_y[i])**2)
                mask[i, ][rr_grid < radius] = 0.

            return mask

        def _create_basis(images, bg_mean, pca_number):
            """
            Method for creating a set of principal components for a stack of images.
            """

            if self.m_subtract_mean:
                images -= bg_mean

            _, _, v_svd = svds(images.reshape(images.shape[0],
                                              images.shape[1]*images.shape[2]),
                               k=pca_number)

            v_svd = v_svd[::-1, ]

            pca_basis = v_svd.reshape(v_svd.shape[0], images.shape[1], images.shape[2])

            return pca_basis

        def _model_background(basis, im_arr, mask):
            """
            Method for creating a model of the background.
            """

            def _dot_product(x_dot, *p):
                return np.dot(p, x_dot)

            fit_im_chi = np.zeros(im_arr.shape)
            # fit_coeff_chi = np.zeros((im_arr.shape[0], basis.shape[0]))

            basis_reshaped = basis.reshape(basis.shape[0], -1)

            for i in range(im_arr.shape[0]):
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

        self.m_residuals_out_port.del_all_data()
        self.m_residuals_out_port.del_all_attributes()

        if self.m_fit_out_port is not None:
            self.m_fit_out_port.del_all_data()
            self.m_fit_out_port.del_all_attributes()

        if self.m_mask_out_port is not None:
            self.m_mask_out_port.del_all_data()
            self.m_mask_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_star_in_port.get_attribute("PIXSCALE")

        nimages = self.m_star_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        self.m_mask_star /= pixscale
        self.m_gaussian = int(math.ceil(self.m_gaussian/pixscale))

        if self.m_subframe is not None:
            self.m_subframe /= pixscale
            self.m_subframe = int(math.ceil(self.m_subframe))

        bg_mean = np.mean(self.m_background_in_port.get_all(), axis=0)

        star = np.zeros((nimages, 2))
        for i, _ in enumerate(star):
            star[i, :] = locate_star(image=self.m_star_in_port[i, ]-bg_mean,
                                     center=None,
                                     width=self.m_subframe,
                                     fwhm=self.m_gaussian)

        if self.m_mask_planet is not None:
            parang = self.m_star_in_port.get_attribute("PARANG")

            self.m_mask_planet = np.asarray(self.m_mask_planet)

            self.m_mask_planet[0] /= pixscale
            self.m_mask_planet[3] /= pixscale

        sys.stdout.write("Creating PCA basis set...")
        sys.stdout.flush()

        basis_pca = _create_basis(self.m_background_in_port.get_all(),
                                  bg_mean,
                                  self.m_pca_number)

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Calculating background model...")

            im_star = self.m_star_in_port[frames[i]:frames[i+1], ]

            if self.m_subtract_mean:
                im_star -= bg_mean

            mask_star = _create_mask(self.m_mask_star,
                                     star[frames[i]:frames[i+1], ],
                                     frames[i+1]-frames[i])

            if self.m_mask_planet is None:
                mask_planet = np.ones(im_star.shape)

            else:
                cent_x = star[frames[i]:frames[i+1], 1]
                cent_y = star[frames[i]:frames[i+1], 0]

                theta = np.radians(self.m_mask_planet[1] + 90. - \
                            parang[frames[i]:frames[i+1]] + self.m_mask_planet[2])

                x_planet = self.m_mask_planet[0]*np.cos(theta) + cent_x
                y_planet = self.m_mask_planet[0]*np.sin(theta) + cent_y

                planet = np.stack((y_planet, x_planet))

                mask_planet = _create_mask(self.m_mask_planet[3],
                                           np.transpose(planet),
                                           frames[i+1]-frames[i])

            fit_im = _model_background(basis_pca,
                                       im_star*mask_star*mask_planet,
                                       mask_star*mask_planet)

            self.m_residuals_out_port.append(im_star-fit_im)

            if self.m_fit_out_port is not None:
                self.m_fit_out_port.append(fit_im)

            if self.m_mask_out_port is not None:
                self.m_mask_out_port.append(mask_star*mask_planet)

        sys.stdout.write("Calculating background model... [DONE]\n")
        sys.stdout.flush()

        history = "PC number = "+str(self.m_pca_number)
        self.m_residuals_out_port.copy_attributes(self.m_star_in_port)
        self.m_residuals_out_port.add_history("PCABackgroundSubtractionModule", history)
        self.m_residuals_out_port.add_attribute("STAR_POSITION", star, static=False)

        if self.m_fit_out_port is not None:
            self.m_fit_out_port.copy_attributes(self.m_star_in_port)
            self.m_fit_out_port.add_history("PCABackgroundSubtractionModule", history)

        if self.m_mask_out_port is not None:
            self.m_mask_out_port.copy_attributes(self.m_star_in_port)
            self.m_mask_out_port.add_history("PCABackgroundSubtractionModule", history)

        self.m_residuals_out_port.close_port()


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
                 size=2.,
                 gaussian=0.15,
                 subframe=None,
                 pca_number=60,
                 mask_star=0.7,
                 subtract_mean=False,
                 **kwargs):
        """
        Constructor of DitheringBackgroundModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Not written if set to None.
        center : tuple(tuple(int, int), )
            Tuple with the centers of the dithering positions, e.g. ((x0,y0), (x1,y1)). The order
            of the coordinates should correspond to the order in which the star is present. If
            *center* and *cubes* are both set to None then sorting and subtracting of the
            background frames is based on DITHER_X and DITHER_Y. If *center* is specified and
            *cubes* is set to None then the DITHER_X and DITHER_Y attributes will be used for
            sorting and subtracting of the background but not for selecting the dither positions.
        cubes : int
            Number of consecutive cubes per dither position. If *cubes* is set to None then sorting
            and subtracting of the background frames is based on DITHER_X and DITHER_Y.
        size : float
            Image size (arsec) that is cropped at the specified dither positions.
        gaussian : float
            Full width at half maximum (arcsec) of the Gaussian kernel that is used to smooth the
            image before the star is located.
        subframe : float
            Size (arcsec) of the subframe that is used to search for the star. Cropping of the
            subframe is done around the center of the dithering position. If set to None then the
            full frame size (*size*) will be used.
        pca_number : int
            Number of principal components.
        mask_star : float
            Radius of the central mask (arcsec).
        subtract_mean : bool
            The mean of the background images is subtracted from both the star and background
            images before the PCA basis is constructed.

        Keyword Arguments
        -----------------
        crop : bool
            Skip the step of selecting and cropping of the dithering positions if set to False.
        prepare : bool
            Skip the step of preparing the PCA background subtraction if set to False.
        pca_background : bool
            Skip the step of the PCA background subtraction if set to False.
        combine : str
            Combine the mean background subtracted ("mean") or PCA background subtracted ("pca")
            frames. This step is ignored if set to None.
        mask_planet : tuple(float, float, float)
            Separation (arcsec), position angle (deg) measured in counterclockwise direction with
            respect to upward direction, additional rotation angle (deg), and radius (arcsec) of
            the mask, (sep, angle, radius). No mask is used when set to None.

        Returns
        -------
        NoneType
            None
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
            self.m_combine = "pca"

        if "mask_planet" in kwargs:
            self.m_mask_planet = kwargs["mask_planet"]
        else:
            self.m_mask_planet = None

        super(DitheringBackgroundModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        self.m_center = center
        self.m_cubes = cubes
        self.m_size = size
        self.m_gaussian = gaussian
        self.m_pca_number = pca_number
        self.m_mask_star = mask_star
        self.m_subtract_mean = subtract_mean
        self.m_subframe = subframe

        self.m_image_in_tag = image_in_tag
        self.m_image_out_tag = image_out_tag

    def _initialize(self):
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
                self.m_center = np.copy(dither) + float(npix)/2.
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

        return n_dither, star_pos

    def run(self):
        """
        Run method of the module. Cuts out the detector sections at the different dither positions,
        prepares the PCA background subtraction, locates the star in each image, runs the PCA
        background subtraction, combines the output from the different dither positions is written
        to a single database tag.

        Returns
        -------
        NoneType
            None
        """

        def _admin_start(count, n_dither, position, star_pos):
            if self.m_crop or self.m_prepare or self.m_pca_background:
                print("Processing dither position "+str(count+1)+" out of "+str(n_dither)+"...")
                print("Center position =", position)

                if self.m_cubes is None and self.m_center is not None:
                    print("DITHER_X, DITHER_Y =", tuple(star_pos))

        def _admin_end(count, n_dither):
            if self.m_combine == "mean":
                tags.append(self.m_image_in_tag+"_dither_mean"+str(count+1))

            elif self.m_combine == "pca":
                tags.append(self.m_image_in_tag+"_dither_pca_res"+str(count+1))

            if self.m_crop or self.m_prepare or self.m_pca_background:
                print("Processing dither position "+str(count+1)+ \
                      " out of "+str(n_dither)+"... [DONE]")

        n_dither, star_pos = self._initialize()
        tags = []

        for i, position in enumerate(self.m_center):
            _admin_start(i, n_dither, position, star_pos[i])

            if self.m_crop:
                module = CropImagesModule(size=self.m_size,
                                          center=(int(math.ceil(position[0])),
                                                  int(math.ceil(position[1]))),
                                          name_in="crop"+str(i),
                                          image_in_tag=self.m_image_in_tag,
                                          image_out_tag=self.m_image_in_tag+ \
                                                        "_dither_crop"+str(i+1))

                module.connect_database(self._m_data_base)
                module.run()

            if self.m_prepare:
                module = PCABackgroundPreparationModule(dither=(n_dither,
                                                                self.m_cubes,
                                                                star_pos[i]),
                                                        name_in="prepare"+str(i),
                                                        image_in_tag=self.m_image_in_tag+ \
                                                                     "_dither_crop"+str(i+1),
                                                        star_out_tag=self.m_image_in_tag+ \
                                                                     "_dither_star"+str(i+1),
                                                        mean_out_tag=self.m_image_in_tag+ \
                                                                     "_dither_mean"+str(i+1),
                                                        background_out_tag=self.m_image_in_tag+ \
                                                                           "_dither_background"+ \
                                                                           str(i+1))

                module.connect_database(self._m_data_base)
                module.run()

            if self.m_pca_background:
                module = PCABackgroundSubtractionModule(pca_number=self.m_pca_number,
                                                        mask_star=self.m_mask_star,
                                                        mask_planet=self.m_mask_planet,
                                                        subtract_mean=self.m_subtract_mean,
                                                        subframe=self.m_subframe,
                                                        name_in="pca_background"+str(i),
                                                        star_in_tag=self.m_image_in_tag+ \
                                                                    "_dither_star"+str(i+1),
                                                        background_in_tag=self.m_image_in_tag+ \
                                                                          "_dither_background"+ \
                                                                          str(i+1),
                                                        residuals_out_tag=self.m_image_in_tag+ \
                                                                          "_dither_pca_res"+ \
                                                                          str(i+1),
                                                        fit_out_tag=self.m_image_in_tag+ \
                                                                    "_dither_pca_fit"+str(i+1),
                                                        mask_out_tag=self.m_image_in_tag+ \
                                                                     "_dither_pca_mask"+str(i+1))

                module.connect_database(self._m_data_base)
                module.run()

            _admin_end(i, n_dither)

        if self.m_combine is not None and self.m_image_out_tag is not None:
            module = CombineTagsModule(name_in="combine",
                                       check_attr=True,
                                       index_init=False,
                                       image_in_tags=tags,
                                       image_out_tag=self.m_image_in_tag+"_dither_combine")

            module.connect_database(self._m_data_base)
            module.run()

            module = SortParangModule(name_in="sort",
                                      image_in_tag=self.m_image_in_tag+"_dither_combine",
                                      image_out_tag=self.m_image_out_tag)

            module.connect_database(self._m_data_base)
            module.run()
