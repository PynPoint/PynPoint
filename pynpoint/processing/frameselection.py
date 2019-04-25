"""
Pipeline modules for frame selection.
"""

from __future__ import absolute_import

import sys
import math
import warnings

import numpy as np

from six.moves import range

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import crop_image, pixel_distance
from pynpoint.util.module import progress, memory_frames, locate_star
from pynpoint.util.remove import write_selected_data, write_selected_attributes


class RemoveFramesModule(ProcessingModule):
    """
    Pipeline module for removing images by their index number.
    """

    def __init__(self,
                 frames,
                 name_in="remove_frames",
                 image_in_tag="im_arr",
                 selected_out_tag="im_arr_selected",
                 removed_out_tag="im_arr_removed"):
        """
        Constructor of RemoveFramesModule.

        Parameters
        ----------
        frames : str, list, tuple, range, or numpy.ndarray
            A tuple or array with the frame indices that have to be removed or a database tag
            pointing to a list of frame indices.
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        selected_out_tag : str
            Tag of the database entry with the remaining images after removing the specified
            images. Should be different from *image_in_tag*. No data is written when set to
            *None*.
        removed_out_tag : str
            Tag of the database entry with the images that are removed. Should be different
            from *image_in_tag*. No data is written when set to *None*.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if selected_out_tag is None:
            self.m_selected_out_port = None
        else:
            self.m_selected_out_port = self.add_output_port(selected_out_tag)

        if removed_out_tag is None:
            self.m_removed_out_port = None
        else:
            self.m_removed_out_port = self.add_output_port(removed_out_tag)

        if isinstance(frames, str):
            self.m_index_in_port = self.add_input_port(frames)
        else:
            self.m_index_in_port = None

            if isinstance(frames, (tuple, list, range)):
                self.m_frames = np.asarray(frames, dtype=np.int)

            elif isinstance(frames, np.ndarray):
                self.m_frames = frames

    def _initialize(self):

        if self.m_selected_out_port is not None:
            if self.m_image_in_port.tag == self.m_selected_out_port.tag:
                raise ValueError("Input and output ports should have a different tag.")

        if self.m_removed_out_port is not None:
            if self.m_image_in_port.tag == self.m_removed_out_port.tag:
                raise ValueError("Input and output ports should have a different tag.")

        if self.m_index_in_port is not None:
            self.m_frames = self.m_index_in_port.get_all()

        if np.size(np.where(self.m_frames >= self.m_image_in_port.get_shape()[0])) > 0:
            raise ValueError("Some values in 'frames' are larger than the total number of "
                             "available frames, %s." % str(self.m_image_in_port.get_shape()[0]))

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.del_all_data()
            self.m_selected_out_port.del_all_attributes()

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.del_all_data()
            self.m_removed_out_port.del_all_attributes()

    def run(self):
        """
        Run method of the module. Removes the frames and corresponding attributes, updates the
        NFRAMES attribute, and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        self._initialize()

        memory = self._m_config_port.get_attribute("MEMORY")

        nimages = self.m_image_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        if memory == 0 or memory >= nimages:
            memory = nimages

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running RemoveFramesModule...")

            images = self.m_image_in_port[frames[i]:frames[i+1], ]

            index_del = np.where(np.logical_and(self.m_frames >= frames[i], \
                                                self.m_frames < frames[i+1]))

            write_selected_data(images,
                                self.m_frames[index_del]%memory,
                                self.m_selected_out_port,
                                self.m_removed_out_port)

        sys.stdout.write("Running RemoveFramesModule... [DONE]\n")
        sys.stdout.flush()

        history = "frames removed = "+str(np.size(self.m_frames))

        if self.m_selected_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_selected_out_port.copy_attributes(self.m_image_in_port)
            self.m_selected_out_port.add_history("RemoveFramesModule", history)

        if self.m_removed_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_removed_out_port.copy_attributes(self.m_image_in_port)
            self.m_removed_out_port.add_history("RemoveFramesModule", history)

        write_selected_attributes(self.m_frames,
                                  self.m_image_in_port,
                                  self.m_selected_out_port,
                                  self.m_removed_out_port)

        self.m_image_in_port.close_port()


class FrameSelectionModule(ProcessingModule):
    """
    Pipeline module for frame selection.
    """

    def __init__(self,
                 name_in="frame_selection",
                 image_in_tag="im_arr",
                 selected_out_tag="im_arr_selected",
                 removed_out_tag="im_arr_removed",
                 index_out_tag=None,
                 method="median",
                 threshold=4.,
                 fwhm=0.1,
                 aperture=("circular", 0.2),
                 position=(None, None, 0.5)):
        """
        Constructor of FrameSelectionModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        selected_out_tag : str
            Tag of the database entry with the selected images that are written as output. Should
            be different from *image_in_tag*. No data is written when set to None.
        removed_out_tag : str
            Tag of the database entry with the removed images that are written as output. Should
            be different from *image_in_tag*. No data is written when set to None.
        index_out_tag : str
            Tag of the database entry with the list of frames indices that are removed with the
            frames selection. No data is written when set to *None*.
        method : str
            Perform the sigma clipping with respect to the median or maximum aperture flux by
            setting the *method* to "median" or "max".
        threshold : float
            Threshold in units of sigma for the frame selection. All images that are a *threshold*
            number of sigmas away from the median photometry will be removed.
        fwhm : float
            The full width at half maximum (FWHM) of the Gaussian kernel (arcsec) that is used to
            smooth the images before the brightest pixel is located. Should be similar in size to
            the FWHM of the stellar PSF. A fixed position, specified by *position*, is used when
            *fwhm* is set to None
        aperture : tuple(str, float, float)
            Tuple with the aperture properties for measuring the photometry around the location of
            the brightest pixel. The first element contains the aperture type ("circular",
            "annulus", or "ratio"). For a circular aperture, the second element contains the
            aperture radius (arcsec). For the other two types, the second and third element are the
            inner and outer radii (arcsec) of the aperture. The position of the aperture has to be
            specified with *position* when *fwhm* is set to None.
        position : tuple(int, int, float)
            Subframe that is selected to search for the star. The tuple contains the center (pix)
            and size (arcsec) (pos_x, pos_y, size). Setting *position* to None will use the full
            image to search for the star. If *position=(None, None, size)* then the center of the
            image will be used.

        Returns
        -------
        NoneType
            None
        """

        super(FrameSelectionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if index_out_tag is None:
            self.m_index_out_port = None
        else:
            self.m_index_out_port = self.add_output_port(index_out_tag)

        if selected_out_tag is None:
            self.m_selected_out_port = None
        else:
            self.m_selected_out_port = self.add_output_port(selected_out_tag)

        if removed_out_tag is None:
            self.m_removed_out_port = None
        else:
            self.m_removed_out_port = self.add_output_port(removed_out_tag)

        self.m_method = method
        self.m_fwhm = fwhm
        self.m_aperture = aperture
        self.m_threshold = threshold
        self.m_position = position

    def _initialize(self):
        if self.m_image_in_port.tag == self.m_selected_out_port.tag or \
                self.m_image_in_port.tag == self.m_removed_out_port.tag:
            raise ValueError("Input and output ports should have a different tag.")

        if self.m_index_out_port is not None:
            self.m_index_out_port.del_all_data()
            self.m_index_out_port.del_all_attributes()

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.del_all_data()
            self.m_selected_out_port.del_all_attributes()

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.del_all_data()
            self.m_removed_out_port.del_all_attributes()

    def run(self):
        """
        Run method of the module. Smooths the images with a Gaussian kernel, locates the brightest
        pixel in each image, measures the integrated flux around the brightest pixel, calculates
        the median and standard deviation of the photometry, and applies sigma clipping to remove
        low quality images.

        Returns
        -------
        NoneType
            None
        """

        def _get_aperture(aperture):
            if aperture[0] == "circular":
                aperture = (0., aperture[1]/pixscale)

            elif aperture[0] == "annulus" or aperture[0] == "ratio":
                aperture = (aperture[1]/pixscale, aperture[2]/pixscale)

            return aperture

        def _get_starpos(fwhm, position):
            starpos = np.zeros((nimages, 2), dtype=np.int64)

            if fwhm is None:
                starpos[:, 0] = position[0]
                starpos[:, 1] = position[1]

            else:
                if position is None:
                    center = None
                    width = None

                else:
                    if position[0] is None and position[1] is None:
                        center = None
                    else:
                        center = position[0:2]

                    width = int(math.ceil(position[2]/pixscale))

                for i, _ in enumerate(starpos):
                    starpos[i, :] = locate_star(image=self.m_image_in_port[i, ],
                                                center=center,
                                                width=width,
                                                fwhm=int(math.ceil(fwhm/pixscale)))

            return starpos

        def _photometry(images, starpos, aperture):
            check_pos_in = any(np.floor(starpos[:]-aperture[1]) < 0.)
            check_pos_out = any(np.ceil(starpos[:]+aperture[1]) > images.shape[0])

            if check_pos_in or check_pos_out:
                phot = np.nan

            else:
                im_crop = crop_image(images, starpos, 2*int(math.ceil(aperture[1])))

                npix = im_crop.shape[0]

                x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)
                xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
                rr_grid = np.sqrt(xx_grid*xx_grid+yy_grid*yy_grid)

                if self.m_aperture[0] == "circular":
                    phot = np.sum(im_crop[rr_grid < aperture[1]])

                elif self.m_aperture[0] == "annulus":
                    phot = np.sum(im_crop[(rr_grid > aperture[0]) & (rr_grid < aperture[1])])

                elif self.m_aperture[0] == "ratio":
                    phot = np.sum(im_crop[rr_grid < aperture[0]]) / \
                        np.sum(im_crop[(rr_grid > aperture[0]) & (rr_grid < aperture[1])])

            return phot

        self._initialize()

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        nimages = self.m_image_in_port.get_shape()[0]

        aperture = _get_aperture(self.m_aperture)
        starpos = _get_starpos(self.m_fwhm, self.m_position)

        phot = np.zeros(nimages)

        for i in range(nimages):
            progress(i, nimages, "Running FrameSelectionModule...")

            images = self.m_image_in_port[i]
            phot[i] = _photometry(images, starpos[i, :], aperture)

        if self.m_method == "median":
            phot_ref = np.nanmedian(phot)
        elif self.m_method == "max":
            phot_ref = np.nanmax(phot)

        phot_std = np.nanstd(phot)

        index_rm = np.logical_or((phot > phot_ref+self.m_threshold*phot_std),
                                 (phot < phot_ref-self.m_threshold*phot_std))

        index_rm[np.isnan(phot)] = True

        indices = np.where(index_rm)[0]
        indices = np.asarray(indices, dtype=np.int)

        if np.size(indices) > 0:
            memory = self._m_config_port.get_attribute("MEMORY")
            frames = memory_frames(memory, nimages)

            if memory == 0 or memory >= nimages:
                memory = nimages

            for i, _ in enumerate(frames[:-1]):
                images = self.m_image_in_port[frames[i]:frames[i+1], ]

                index_del = np.where(np.logical_and(indices >= frames[i], \
                                                    indices < frames[i+1]))

                write_selected_data(images,
                                    indices[index_del]%memory,
                                    self.m_selected_out_port,
                                    self.m_removed_out_port)

        else:
            warnings.warn("No frames were removed.")

        history = "frames removed = "+str(np.size(indices))

        if self.m_index_out_port is not None:
            self.m_index_out_port.set_all(np.transpose(indices))
            self.m_index_out_port.copy_attributes(self.m_image_in_port)
            self.m_index_out_port.add_attribute("STAR_POSITION", starpos, static=False)
            self.m_index_out_port.add_history("FrameSelectionModule", history)

        if self.m_selected_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_selected_out_port.copy_attributes(self.m_image_in_port)

        if self.m_removed_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_removed_out_port.copy_attributes(self.m_image_in_port)

        write_selected_attributes(indices,
                                  self.m_image_in_port,
                                  self.m_selected_out_port,
                                  self.m_removed_out_port)

        if self.m_selected_out_port is not None:
            indices_select = np.ones(nimages, dtype=bool)
            indices_select[indices] = False
            indices_select = np.where(indices_select)

            self.m_selected_out_port.add_attribute("STAR_POSITION",
                                                   starpos[indices_select],
                                                   static=False)

            self.m_selected_out_port.add_history("FrameSelectionModule", history)

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.add_attribute("STAR_POSITION",
                                                  starpos[indices],
                                                  static=False)

            self.m_removed_out_port.add_history("FrameSelectionModule", history)

        sys.stdout.write("Running FrameSelectionModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_in_port.close_port()


class RemoveLastFrameModule(ProcessingModule):
    """
    Pipeline module for removing every NDIT+1 frame from NACO data obtained in cube mode. This
    frame contains the average pixel values of the cube.
    """

    def __init__(self,
                 name_in="remove_last_frame",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_last"):
        """
        Constructor of RemoveLastFrameModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveLastFrameModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Removes every NDIT+1 frame and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        ndit = self.m_image_in_port.get_attribute("NDIT")
        nframes = self.m_image_in_port.get_attribute("NFRAMES")
        index = self.m_image_in_port.get_attribute("INDEX")

        nframes_new = []
        index_new = []

        for i, item in enumerate(ndit):
            progress(i, len(ndit), "Running RemoveLastFrameModule...")

            if nframes[i] != item+1:
                warnings.warn("Number of frames (%s) is not equal to NDIT+1." % nframes[i])

            frame_start = np.sum(nframes[0:i])
            frame_end = np.sum(nframes[0:i+1]) - 1

            nframes_new.append(nframes[i]-1)
            index_new.extend(index[frame_start:frame_end])

            images = self.m_image_in_port[frame_start:frame_end, ]
            self.m_image_out_port.append(images)

        nframes_new = np.asarray(nframes_new, dtype=np.int)
        index_new = np.asarray(index_new, dtype=np.int)

        sys.stdout.write("Running RemoveLastFrameModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.add_attribute("NFRAMES", nframes_new, static=False)
        self.m_image_out_port.add_attribute("INDEX", index_new, static=False)

        history = "frames removed = NDIT+1"
        self.m_image_out_port.add_history("RemoveLastFrameModule", history)

        self.m_image_out_port.close_port()


class RemoveStartFramesModule(ProcessingModule):
    """
    Pipeline module for removing a fixed number of images at the beginning of each cube. This can
    be useful for NACO data in which the background is significantly higher in the first several
    frames of a data cube.
    """

    def __init__(self,
                 frames=1,
                 name_in="remove_last_frame",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_first"):
        """
        Constructor of RemoveStartFramesModule.

        Parameters
        ----------
        frames : int
            Number of frames that are removed at the beginning of each cube.
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveStartFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_frames = int(frames)

    def run(self):
        """
        Run method of the module. Removes a constant number of images at the beginning of each cube
        and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        nframes = self.m_image_in_port.get_attribute("NFRAMES")
        index = self.m_image_in_port.get_attribute("INDEX")

        index_new = []

        if "PARANG" in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute("PARANG")
            parang_new = []

        else:
            parang = None

        if "STAR_POSITION" in self.m_image_in_port.get_all_non_static_attributes():
            star = self.m_image_in_port.get_attribute("STAR_POSITION")
            star_new = []

        else:
            star = None

        for i, _ in enumerate(nframes):
            progress(i, len(nframes), "Running RemoveStartFramesModule...")

            frame_start = np.sum(nframes[0:i]) + self.m_frames
            frame_end = np.sum(nframes[0:i+1])

            if frame_start >= frame_end:
                raise ValueError("The number of frames in the original data cube is equal or "
                                 "smaller than the number of frames that have to be removed.")

            index_new.extend(index[frame_start:frame_end])

            if parang is not None:
                parang_new.extend(parang[frame_start:frame_end])

            if star is not None:
                star_new.extend(star[frame_start:frame_end])

            images = self.m_image_in_port[frame_start:frame_end, ]
            self.m_image_out_port.append(images)

        sys.stdout.write("Running RemoveStartFramesModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.add_attribute("NFRAMES", nframes-self.m_frames, static=False)
        self.m_image_out_port.add_attribute("INDEX", index_new, static=False)

        if parang is not None:
            self.m_image_out_port.add_attribute("PARANG", parang_new, static=False)

        if star is not None:
            self.m_image_out_port.add_attribute("STAR_POSITION", np.asarray(star_new), static=False)

        history = "frames removed = "+str(self.m_frames)
        self.m_image_out_port.add_history("RemoveStartFramesModule", history)

        self.m_image_out_port.close_port()


class ImageStatisticsModule(ProcessingModule):
    """
    Pipeline module for calculating image statistics for the full images or a subsection of the
    images.
    """

    def __init__(self,
                 name_in="im_stat",
                 image_in_tag="im_arr",
                 stat_out_tag="stat",
                 position=None):
        """
        Constructor of ImageStatisticsModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the images that are read as input.
        stat_out_tag : str
            Tag of the database entry with the statistical results that are written as output. The
            result is stored in the following order: minimum, maximum, sum, mean, median, and
            standard deviation.
        position : tuple(int, int, float)
            Position (x, y) (pix) and radius (arcsec) of the circular area in which the statistics
            are calculated. The full image is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        super(ImageStatisticsModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_stat_out_port = self.add_output_port(stat_out_tag)

        self.m_position = position

    def run(self):
        """
        Run method of the module. Calculates the minimum, maximum, sum, mean, median, and standard
        deviation of the pixel values of each image separately. NaNs are ignored for each
        calculation. The values are calculated for either the full images or a circular
        subsection of the images.

        Returns
        -------
        NoneType
            None
        """

        self.m_stat_out_port.del_all_data()
        self.m_stat_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if self.m_position is not None:
            self.m_position = (int(self.m_position[1]), # y position
                               int(self.m_position[0]), # x position
                               self.m_position[2]/pixscale) # radius (pix)

        nimages = self.m_image_in_port.get_shape()[0]
        im_shape = self.m_image_in_port.get_shape()[1:]

        frames = memory_frames(memory, nimages)

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running ImageStatisticsModule...")

            images = self.m_image_in_port[frames[i]:frames[i+1], ]
            images = np.reshape(images, (images.shape[0], im_shape[0]*im_shape[1]))

            if self.m_position is not None:
                rr_grid = pixel_distance(im_shape, self.m_position)
                indices = np.where(rr_grid <= self.m_position[2])[0]
                images = images[:, indices]

            nmin = np.nanmin(images, axis=1)
            nmax = np.nanmax(images, axis=1)
            nsum = np.nansum(images, axis=1)
            mean = np.nanmean(images, axis=1)
            median = np.nanmedian(images, axis=1)
            std = np.nanstd(images, axis=1)

            result = np.column_stack((nmin, nmax, nsum, mean, median, std))
            self.m_stat_out_port.append(result)

        sys.stdout.write("Running ImageStatisticsModule... [DONE]\n")
        sys.stdout.flush()

        history = "number of images = "+str(nimages)
        self.m_stat_out_port.copy_attributes(self.m_image_in_port)
        self.m_stat_out_port.add_history("ImageStatisticsModule", history)
        self.m_stat_out_port.close_port()
