"""
Modules with tools for frame selection.
"""

import sys
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
                 selected_out_tag="im_arr_selected",
                 removed_out_tag="im_arr_removed"):
        """
        Constructor of RemoveFramesModule.

        :param frames: Frame indices to be removed. Python indexing starts at 0.
        :type frames: tuple or array, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param selected_out_tag: Tag of the database entry with the selected images that are
                                 written as output. Should be different from *image_in_tag*.
                                 No data is written when set to *None*.
        :type selected_out_tag: str
        :param removed_out_tag: Tag of the database entry with the removed images that are
                                written as output. Should be different from *image_in_tag*.
                                No data is written when set to *None*.
        :type removed_out_tag: str

        :return: None
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

        self.m_frames = np.asarray(frames, dtype=np.int)

    def run(self):
        """
        Run method of the module. Removes the frames and corresponding attributes, updates the
        NFRAMES attribute, and saves the data and attributes.

        :return: None
        """

        if self.m_selected_out_port is not None:
            if self.m_image_in_port.tag == self.m_selected_out_port.tag:
                raise ValueError("Input and output ports should have a different tag.")

        if self.m_removed_out_port is not None:
            if self.m_image_in_port.tag == self.m_removed_out_port.tag:
                raise ValueError("Input and output ports should have a different tag.")

        if np.size(np.where(self.m_frames >= self.m_image_in_port.get_shape()[0])) > 0:
            raise ValueError("Some values in frames are larger than the total number of "
                             "available frames, %s." % str(self.m_image_in_port.get_shape()[0]))

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.del_all_data()
            self.m_selected_out_port.del_all_attributes()

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.del_all_data()
            self.m_removed_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")
        nimages = self.m_image_in_port.get_shape()[0]

        if memory == 0 or memory >= nimages:
            frames = [0, nimages]
            memory = nimages

        else:
            frames = np.linspace(0,
                                 nimages-nimages%memory,
                                 int(float(nimages)/float(memory))+1,
                                 endpoint=True,
                                 dtype=np.int)

            if nimages%memory > 0:
                frames = np.append(frames, nimages)

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running RemoveFramesModule...")

            images = self.m_image_in_port[frames[i]:frames[i+1], ]

            index_del = np.where(np.logical_and(self.m_frames >= frames[i], \
                                                self.m_frames < frames[i+1]))

            if np.size(index_del) > 0:
                if self.m_removed_out_port is not None:
                    self.m_removed_out_port.append(images[self.m_frames[index_del]%memory])
                images = np.delete(images, self.m_frames[index_del]%memory, axis=0)

            if self.m_selected_out_port is not None:
                self.m_selected_out_port.append(images)

        sys.stdout.write("Running RemoveFramesModule... [DONE]\n")
        sys.stdout.flush()

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        non_static = self.m_image_in_port.get_all_non_static_attributes()

        index = self.m_image_in_port.get_attribute("INDEX")

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.add_attribute("INDEX",
                                                   np.delete(index, self.m_frames),
                                                   static=False)

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.add_attribute("INDEX",
                                                  index[self.m_frames],
                                                  static=False)

        if "PARANG" in non_static:
            parang = self.m_image_in_port.get_attribute("PARANG")

            if self.m_selected_out_port is not None:
                self.m_selected_out_port.add_attribute("PARANG",
                                                       np.delete(parang, self.m_frames),
                                                       static=False)

            if self.m_removed_out_port is not None:
                self.m_removed_out_port.add_attribute("PARANG",
                                                      parang[self.m_frames],
                                                      static=False)

        if "STAR_POSITION" in non_static:
            position = self.m_image_in_port.get_attribute("STAR_POSITION")

            if self.m_selected_out_port is not None:
                self.m_selected_out_port.add_attribute("STAR_POSITION",
                                                       np.delete(position, self.m_frames, axis=0),
                                                       static=False)

            if self.m_removed_out_port is not None:
                self.m_removed_out_port.add_attribute("STAR_POSITION",
                                                      position[self.m_frames],
                                                      static=False)

        if "NFRAMES" in non_static:
            nframes = self.m_image_in_port.get_attribute("NFRAMES")

            nframes_sel = np.zeros(nframes.shape)
            nframes_del = np.zeros(nframes.shape)

            for i, frames in enumerate(nframes):
                total = np.sum(nframes[0:i])

                index_del = np.where(np.logical_and(self.m_frames >= total, \
                                     self.m_frames < total+frames))[0]

                nframes_sel[i] = frames-np.size(index_del)
                nframes_del[i] = np.size(index_del)

            if self.m_selected_out_port is not None:
                self.m_selected_out_port.add_attribute("NFRAMES", nframes_sel, static=False)

            if self.m_removed_out_port is not None:
                self.m_removed_out_port.add_attribute("NFRAMES", nframes_del, static=False)

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.add_history_information("Frames removed",
                                                             str(np.size(self.m_frames)))

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.add_history_information("Frames removed",
                                                            str(np.size(self.m_frames)))

        self.m_image_in_port.close_port()


class FrameSelectionModule(ProcessingModule):
    """
    Module for frame selection.
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
                 aperture=0.2,
                 position=(None, None, 0.5)):
        """
        Constructor of FrameSelectionModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param selected_out_tag: Tag of the database entry with the selected images that are
                                 written as output. Should be different from *image_in_tag*.
                                 No data is written when set to *None*.
        :type selected_out_tag: str
        :param removed_out_tag: Tag of the database entry with the removed images that are
                                written as output. Should be different from *image_in_tag*.
                                No data is written when set to *None*.
        :type removed_out_tag: str
        :param index_out_tag: Tag of the database entry with the list of frames indices that are
                              removed with the frames selection. No data is written when set to
                              *None*.
        :type index_out_tag: str
        :param method: Perform the sigma clipping with respect to the median or maximum aperture
                       flux by setting the method to *median* or *max*, respectively.
        :type method: str
        :param threshold: Threshold in units of sigma for the frame selection. All images that
                          are a *threshold* number of sigmas away from the median photometry will
                          be removed.
        :type threshold: float
        :param fwhm: The full width at half maximum (FWHM) of the Gaussian kernel (arcsec) that is
                     used to smooth the images before the brightest pixel is located. Should be
                     similar in size to the FWHM of the stellar PSF. A fixed position, specified
                     by *position*, is used when *fwhm* is set to *None*.
        :type fwhm: float
        :param aperture: The aperture radius (arcsec) that is used for measuring the photometry
                         around the location of the brightest pixel. Typically a few times the
                         stellar FWHM would be recommended. The position of the aperture has to
                         be specified with *position* when *fwhm=None*.
        :type aperture: float
        :param position: Subframe that is selected to search for the star. The tuple can contain a
                         single position (pix) and size (arcsec) as (pos_x, pos_y, size), or the
                         position and size can be defined for each image separately in which case
                         the tuple should be 2D (nimages x 3). Setting *position* to None will use
                         the full image to search for the star. If *position=(None, None, size)* then
                         the center of the image will be used. The value of *size* is not used when
                         *fwhm=None*.
        :type position: tuple, float

        :return: None
        """

        super(FrameSelectionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        if index_out_tag is None:
            self.m_index_out_port = None
        else:
            self.m_index_out_port = self.add_output_port(index_out_tag)

        self.m_selected_out_tag = selected_out_tag
        self.m_removed_out_tag = removed_out_tag

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

        if self.m_image_in_port.tag == self.m_selected_out_tag or \
                self.m_image_in_port.tag == self.m_removed_out_tag:
            raise ValueError("Input and output ports should have a different tag.")

        if self.m_index_out_port is not None:
            self.m_index_out_port.del_all_data()
            self.m_index_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_aperture /= pixscale

        nimages = self.m_image_in_port.get_shape()[0]
        npix = self.m_image_in_port.get_shape()[1]

        if self.m_position is None:
            self.m_position = (float(npix)/2., float(npix)/2., None)

        elif self.m_position[0] is None and self.m_position[1] is None:
            self.m_position = (float(npix)/2., float(npix)/2., self.m_position[2])

        if memory == 0 or memory >= nimages:
            frames = [0, nimages]

        else:
            frames = np.linspace(0,
                                 nimages-nimages%memory,
                                 int(float(nimages)/float(memory))+1,
                                 endpoint=True,
                                 dtype=np.int)

            if nimages%memory > 0:
                frames = np.append(frames, nimages)

        phot = np.zeros(nimages)

        if self.m_fwhm is None:
            starpos = np.zeros((nimages, 2), dtype=np.int64)
            starpos[:, 0] = self.m_position[0]
            starpos[:, 1] = self.m_position[1]

        else:
            star = StarExtractionModule(name_in="star",
                                        image_in_tag=self.m_image_in_port.tag,
                                        image_out_tag=None,
                                        image_size=None,
                                        fwhm_star=self.m_fwhm,
                                        position=self.m_position)

            star.connect_database(self._m_data_base)
            star.run()

            starpos = self.m_image_in_port.get_attribute("STAR_POSITION")

        rr_grid = None

        for i in range(nimages):
            progress(i, nimages, "Running FrameSelectionModule...")

            im_smooth = self.m_image_in_port[i]

            check_pos_in = any(np.floor(starpos[i, :]-self.m_aperture) < 0.)
            check_pos_out = any(np.ceil(starpos[i, :]+self.m_aperture) > im_smooth.shape[0])

            if check_pos_in or check_pos_out:
                phot[i] = np.nan

            else:
                im_cut = Cutout2D(im_smooth,
                                  (starpos[i, 1], starpos[i, 0]),
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
        indices = np.where(index_rm)

        sys.stdout.write("Running FrameSelectionModule... [DONE]\n")
        sys.stdout.flush()

        if np.size(indices) > 0:
            if self.m_index_out_port is not None:
                self.m_index_out_port.set_all(indices)
                self.m_index_out_port.copy_attributes_from_input_port(self.m_image_in_port)
                self.m_index_out_port.add_history_information("Frames removed",
                                                              str(np.size(indices)))

            if self.m_selected_out_tag is not None or self.m_removed_out_tag is not None:
                remove = RemoveFramesModule(frames=indices,
                                            name_in="remove",
                                            image_in_tag=self.m_image_in_port.tag,
                                            selected_out_tag=self.m_selected_out_tag,
                                            removed_out_tag=self.m_removed_out_tag)

                remove.connect_database(self._m_data_base)
                remove.run()

        else:
            print "No frames where removed with the frame selection. [WARNING]"

        self.m_image_in_port.close_port()


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
        index = self.m_image_in_port.get_attribute("INDEX")

        nframes_new = []
        index_new = []

        for i, item in enumerate(ndit):
            progress(i, len(ndit), "Running RemoveLastFrameModule...")

            frame_start = np.sum(nframes[0:i])

            if nframes[i] == item+1:
                frame_end = np.sum(nframes[0:i+1]) - 1
                nframes_new.append(nframes[i]-1)

            else:
                frame_end = np.sum(nframes[0:i+1])
                nframes_new.append(nframes[i])

                warnings.warn("Number of frames (%s) is smaller than NDIT+1." % nframes[i])

            index_new.extend(index[frame_start:frame_end])

            images = self.m_image_in_port[frame_start:frame_end, ]
            self.m_image_out_port.append(images)

        sys.stdout.write("Running RemoveLastFrameModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_attribute("NFRAMES", nframes_new, static=False)
        self.m_image_out_port.add_attribute("INDEX", index_new, static=False)
        self.m_image_out_port.add_history_information("Frames removed", "NDIT+1")

        self.m_image_out_port.close_port()


class RemoveStartFramesModule(ProcessingModule):
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
        Constructor of RemoveStartFramesModule.

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

        super(RemoveStartFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_frames = int(frames)

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
        index = self.m_image_in_port.get_attribute("INDEX")

        index_new = []

        if "PARANG" in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute("PARANG")
            parang_new = []

        else:
            parang = None

        for i, _ in enumerate(nframes):
            progress(i, len(nframes), "Running RemoveStartFramesModule...")

            frame_start = np.sum(nframes[0:i]) + self.m_frames
            frame_end = np.sum(nframes[0:i+1])

            index_new.extend(index[frame_start:frame_end])
            if parang is not None:
                parang_new.extend(parang[frame_start:frame_end])

            images = self.m_image_in_port[frame_start:frame_end, ]
            self.m_image_out_port.append(images)

        sys.stdout.write("Running RemoveStartFramesModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_attribute("NFRAMES", nframes-self.m_frames, static=False)
        self.m_image_out_port.add_attribute("INDEX", index_new, static=False)
        if parang is not None:
            self.m_image_out_port.add_attribute("PARANG", parang_new, static=False)

        self.m_image_out_port.add_history_information("Frames removed", str(self.m_frames))
        self.m_image_out_port.close_port()
