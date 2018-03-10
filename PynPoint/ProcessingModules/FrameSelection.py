"""
Modules with tools for frame selection.
"""

import sys
import math
import warnings

import numpy as np

from astropy.nddata import Cutout2D

from PynPoint.Core.Processing import ProcessingModule
from PynPoint.ProcessingModules.StarAlignment import StarExtractionModule
from PynPoint.Util.Progress import progress


class RemoveFramesModule(ProcessingModule):
    """
    Module for removing frames.
    """

    def __init__(self,
                 frames,
                 name_in="remove_frames",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_remove"):
        """
        Constructor of RemoveFramesModule.

        :param frames: Frame indices to be removed. Python indexing starts at 0.
        :type frames: tuple or array, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(RemoveFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_frames = np.asarray(frames)

    def run(self):
        """
        Run method of the module. Removes the frames, removes the associated NEW_PARA values,
        updates the NAXIS3 value, and saves the data and attributes.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        if np.size(np.where(self.m_frames >= self.m_image_in_port.get_shape()[0])) > 0:
            raise ValueError("Some values in frames are larger than the total number of "
                             "available frames, %s." % str(self.m_image_in_port.get_shape()[0]))

        nframes = self.m_image_in_port.get_shape()[0]
        nstacks = int(float(nframes)/float(memory))

        for i in range(nstacks):
            progress(i, nstacks, "Running RemoveFramesModule...")

            tmp_im = self.m_image_in_port[i*memory:(i+1)*memory, ]

            index_del = np.where(np.logical_and(self.m_frames >= i*memory, \
                                 self.m_frames < (i+1)*memory))

            if np.size(index_del) > 0:
                tmp_im = np.delete(tmp_im,
                                   self.m_frames[index_del]%memory,
                                   axis=0)

            self.m_image_out_port.append(tmp_im)

        sys.stdout.write("Running RemoveFramesModule... [DONE]\n")
        sys.stdout.flush()

        index_del = np.where(self.m_frames >= nstacks*memory)[0]

        if np.size(index_del) > 0:
            tmp_im = self.m_image_in_port[nstacks*memory: \
                                          self.m_image_in_port.get_shape()[0], ]

            tmp_im = np.delete(tmp_im,
                               self.m_frames[index_del]%memory,
                               axis=0)

            self.m_image_out_port.append(tmp_im)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        if "NEW_PARA" in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute("NEW_PARA")
            self.m_image_out_port.add_attribute("NEW_PARA",
                                                np.delete(parang, self.m_frames),
                                                static=False)

        if "STAR_POSITION" in self.m_image_in_port.get_all_non_static_attributes():
            position = self.m_image_in_port.get_attribute("STAR_POSITION")
            self.m_image_out_port.add_attribute("STAR_POSITION",
                                                np.delete(position, self.m_frames, axis=0),
                                                static=False)

        nframes_in = self.m_image_in_port.get_attribute("NFRAMES")
        nframes_out = np.copy(nframes_in)

        total = 0
        for i, frames in enumerate(nframes_in):
            index_del = np.where(np.logical_and(self.m_frames >= total, \
                                 self.m_frames < total+frames))[0]

            nframes_out[i] -= np.size(index_del)

            total += frames

        self.m_image_out_port.add_attribute("NFRAMES", nframes_out, static=False)
        self.m_image_out_port.add_history_information("Frames removed",
                                                      str(np.size(self.m_frames)))
        self.m_image_in_port.close_database()


class FrameSelectionModule(ProcessingModule):
    """
    Module for frame selection.
    """

    def __init__(self,
                 name_in="frame_selection",
                 image_in_tag="im_arr",
                 selected_out_tag="im_arr_selected",
                 removed_out_tag="im_arr_removed",
                 method="median",
                 threshold=4.,
                 fwhm=0.2,
                 aperture=0.5,
                 position=None):
        """
        Constructor of FrameSelectionModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param selected_out_tag: Tag of the database entry with the selected images that are
                                 written as output. Should be different from *image_in_tag*.
        :type selected_out_tag: str
        :param removed_out_tag: Tag of the database entry with the removed images that are
                                written as output. Should be different from *image_in_tag*.
        :type removed_out_tag: str
        :param method: Perform the sigma clipping with respect to the median or maximum aperture
                       flux by setting the method to *median* or *max*, respectively.
        :type method: str
        :param threshold: Threshold in units of sigma for the frame selection. All images that
                          are a *threshold* number of sigmas away from the median photometry will
                          be removed.
        :type threshold: float
        :param fwhm: The full width at half maximum (FWHM) of the Gaussian kernel (arcsec) that is
                     used to smooth the images before the brightest pixel is located. Should be
                     similar in size to the FWHM of the stellar PSF.
        :type fwhm: float
        :param aperture: The aperture radius (arcsec) that is used for measuring the photometry
                         around the location of the brightest pixel. Typically a few times the
                         stellar FWHM would be recommended.
        :type aperture: float
        :param position: Subframe that is selected to search for the star. The tuple can contain a
                         single position and size as (pos_x, pos_y, size), or the position and size
                         can be defined for each image separately in which case the tuple should be
                         2D (nframes x 3). Setting *position* to None will use the full image to
                         search for the star. If position=(None, None, size) then the center of the
                         image will be used.
        :type position: tuple, float

        :return: None
        """

        super(FrameSelectionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_selected_out_port = self.add_output_port(selected_out_tag)
        self.m_removed_out_port = self.add_output_port(removed_out_tag)

        self.m_method = method
        self.m_fwhm = fwhm
        self.m_aperture = aperture
        self.m_threshold = threshold
        self.m_position = position

    def run(self):
        """
        Run method of the module. Smooths the images with a Gaussian kernel, locates the brightest
        pixel in each image, measures the integrated flux around the brightest pixel, calculates
        the median and standard deviation of the photometry, and applies sigma clipping to remove
        images that are of poor quality (e.g., due to opening of the AO loop).

        :return: None
        """

        if self.m_image_in_port.tag == self.m_selected_out_port.tag or \
                self.m_image_in_port.tag == self.m_removed_out_port.tag:
            raise ValueError("Input and output ports should have a different tag.")

        self.m_selected_out_port.del_all_data()
        self.m_selected_out_port.del_all_attributes()

        self.m_removed_out_port.del_all_data()
        self.m_removed_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if "NEW_PARA" in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute("NEW_PARA")
        else:
            parang = None

        self.m_aperture /= pixscale

        nframes = self.m_image_in_port.get_shape()[0]
        if memory == -1 or memory >= nframes:
            frames = [0, nframes]
        else:
            frames = [0]
            for i in range(int(float(nframes)/float(memory))):
                frames.append((i+1)*memory)
            if frames[-1] != nframes:
                frames.append(nframes)

        position = np.zeros((nframes, 2), dtype=np.int64)
        phot = np.zeros(nframes)

        star = StarExtractionModule(name_in="star",
                                    image_in_tag=self.m_image_in_port.tag,
                                    image_out_tag=None,
                                    image_size=None,
                                    fwhm_star=self.m_fwhm,
                                    position=self.m_position)

        star.connect_database(self._m_data_base)
        star.run()

        position = self.m_image_in_port.get_attribute("STAR_POSITION")

        rr_grid = None

        for i in range(nframes):
            progress(i, nframes+nframes, "Running FrameSelectionModule...")

            im_smooth = self.m_image_in_port[i]

            check_pos_in = any(np.floor(position[i, :]-self.m_aperture) < 0.)
            check_pos_out = any(np.ceil(position[i, :]+self.m_aperture) > im_smooth.shape[0])

            if check_pos_in or check_pos_out:
                phot[i] = np.nan

            else:
                im_cut = Cutout2D(im_smooth,
                                  (position[i, 1], position[i, 0]),
                                  size=2.*self.m_aperture).data

                if rr_grid is None:
                    npix = im_cut.shape[0]

                    if npix%2 == 0:
                        x_grid = y_grid = np.linspace(-npix/2+0.5, npix/2-0.5, npix)
                    elif npix%2 == 1:
                        x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)

                    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
                    rr_grid = np.sqrt(xx_grid*xx_grid+yy_grid*yy_grid)

                im_cut[rr_grid >= self.m_aperture] = 0.

                phot[i] = np.sum(im_cut)

        if self.m_method == "median":
            phot_ref = np.nanmedian(phot)
        elif self.m_method == "max":
            phot_ref = np.nanmax(phot)
        else:
            raise ValueError("The method argument should be set to 'median' or 'max'.")

        phot_std = np.nanstd(phot)

        index_rm = np.logical_or((phot > phot_ref+self.m_threshold*phot_std),
                                 (phot < phot_ref-self.m_threshold*phot_std))

        index_rm[np.isnan(phot)] = True

        for i, item in enumerate(frames[:-1]):
            progress(nframes+item, nframes+nframes, "Running FrameSelectionModule...")

            index = index_rm[frames[i]:frames[i+1], ]
            image = self.m_image_in_port[frames[i]:frames[i+1], ]

            if np.size(image[np.logical_not(index)]) > 0:
                self.m_selected_out_port.append(image[np.logical_not(index)])
            if np.size(image[index]) > 0:
                self.m_removed_out_port.append(image[index])

        sys.stdout.write("Running FrameSelectionModule... [DONE]\n")
        sys.stdout.flush()

        nframes_in = self.m_image_in_port.get_attribute("NFRAMES")

        nframes_del = np.zeros(np.size(nframes_in), dtype=np.int64)
        nframes_sel = np.zeros(np.size(nframes_in), dtype=np.int64)

        total = 0
        for i, frames in enumerate(nframes_in):
            nframes_del[i] = np.size(np.where(index_rm[total:total+frames])[0])
            nframes_sel[i] = frames - nframes_del[i]
            total += frames

        n_rm = np.size(index_rm[index_rm])

        self.m_selected_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        if parang is not None:
            self.m_selected_out_port.add_attribute("NEW_PARA",
                                                   parang[np.logical_not(index_rm)],
                                                   static=False)

        if "STAR_POSITION" in self.m_image_in_port.get_all_non_static_attributes():
            self.m_selected_out_port.add_attribute("STAR_POSITION",
                                                   position[np.logical_not(index_rm)],
                                                   static=False)

        self.m_selected_out_port.add_attribute("NFRAMES",
                                               nframes_sel,
                                               static=False)

        self.m_selected_out_port.add_history_information("Frame selection",
                                                         str(n_rm)+" images removed")

        if np.size(index_rm[index_rm]) > 0:

            self.m_removed_out_port.copy_attributes_from_input_port(self.m_image_in_port)

            if parang is not None:
                self.m_removed_out_port.add_attribute("NEW_PARA",
                                                      parang[index_rm],
                                                      static=False)

            if "STAR_POSITION" in self.m_image_in_port.get_all_non_static_attributes():
                self.m_removed_out_port.add_attribute("STAR_POSITION",
                                                      position[index_rm],
                                                      static=False)

            self.m_removed_out_port.add_attribute("NFRAMES",
                                                  nframes_del,
                                                  static=False)

            self.m_removed_out_port.add_history_information("Frame selection",
                                                            str(n_rm)+" images removed")

        else:
            warnings.warn("No frames were removed.")

        self.m_image_in_port.close_database()


class RemoveLastFrameModule(ProcessingModule):
    """
    Module for removing every NDIT+1 frame from NACO data obtained in cube mode. This frame contains
    the average pixel values of the cube.
    """

    def __init__(self,
                 name_in="remove_last_frame",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_last"):
        """
        Constructor of RemoveLastFrameModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(RemoveLastFrameModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Removes every NDIT+1 frame and saves the data and attributes.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        ndit = self.m_image_in_port.get_attribute("NDIT")
        nframes = self.m_image_in_port.get_attribute("NFRAMES")

        for i, item in enumerate(ndit):
            progress(i, len(ndit), "Running RemoveLastFrameModule...")

            if nframes[i] == item+1:
                im_in = self.m_image_in_port[np.sum(nframes[0:i]):np.sum(nframes[0:i+1]), ]
                im_out = np.delete(im_in, nframes[i]-1, axis=0)

                self.m_image_out_port.append(im_out)

            else:
                warnings.warn("Number of frames (%s) is smaller than NDIT+1." % nframes[i])

        sys.stdout.write("Running RemoveLastFrameModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_attribute("NFRAMES", nframes-1, static=False)
        self.m_image_out_port.add_history_information("Frames removed", "NDIT+1")
        self.m_image_out_port.close_database()


class RemoveFirstFrameModule(ProcessingModule):
    """
    Module for removing a fixed number of images at the beginning of each cube. This can be 
    useful for NACO data in which the background is significantly higher in the first several
    frames of a data cube.
    """

    def __init__(self,
                 frames=1,
                 name_in="remove_last_frame",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_first"):
        """
        Constructor of RemoveFirstFrameModule.

        :param frames: Number of frames that are removed at the beginning of each cube.
        :type frames: int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(RemoveFirstFrameModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_frames = frames

    def run(self):
        """
        Run method of the module. Removes a constant number of images at the beginning of each cube
        and saves the data and attributes.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        nframes = self.m_image_in_port.get_attribute("NFRAMES")

        for i, item in enumerate(nframes):
            progress(i, len(nframes), "Running RemoveFirstFrameModule...")

            frame_start = np.sum(nframes[0:i]) + self.m_frames
            frame_end = np.sum(nframes[0:i+1])

            images = self.m_image_in_port[frame_start:frame_end, ]
            self.m_image_out_port.append(images)

        sys.stdout.write("Running RemoveLastFrameModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_attribute("NFRAMES", nframes-self.m_frames, static=False)
        self.m_image_out_port.add_history_information("Frames removed", str(self.m_frames))
        self.m_image_out_port.close_database()
