"""
Modules to prepare the data for the PSF subtraction.
"""

from __future__ import division
from __future__ import absolute_import

import sys
import warnings

import ephem
import numpy as np

from scipy import ndimage
from six.moves import range

from PynPoint.Core.Processing import ProcessingModule
from PynPoint.Util.ModuleTools import progress, memory_frames, number_images_port
from PynPoint.Util.ImageTools import create_mask, scale_image, shift_image


class PSFpreparationModule(ProcessingModule):
    """
    Module to prepare the data for PSF subtraction with PCA. The preparation steps include
    resizing, masking, and image normalization.
    """

    def __init__(self,
                 name_in="psf_preparation",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr",
                 mask_out_tag="mask_arr",
                 norm=True,
                 resize=None,
                 cent_size=None,
                 edge_size=None):
        """
        Constructor of PSFpreparationModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry with images that is written as output.
        :type image_out_tag: str
        :param mask_out_tag: Tag of the database entry with the mask that is written as output.
        :type mask_out_tag: str
        :param norm: Normalization of each image by its Frobenius norm.
        :type norm: bool
        :param resize: Factor by which the data is resized. For example, if *resize* is 2 then
                       the data will be upsampled by a factor of two. No resizing is applied
                       when set to None.
        :type resize: float
        :param cent_size: Radius of the central mask (arcsec). No mask is used when set to None.
        :type cent_size: float
        :param edge_size: Outer radius (arcsec) beyond which pixels are masked. No outer mask is
                          used when set to None. If the value is larger than half the image size
                          then it will be set to half the image size.
        :type edge_size: float

        :return: None
        """

        super(PSFpreparationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if mask_out_tag is not None:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)
        else:
            self.m_mask_out_port = None

        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_resize = resize
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_norm = norm

    def run(self):
        """
        Run method of the module. Normalizes, resizes, and masks the images.

        :return: None
        """

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale

        if self.m_edge_size is not None:
            self.m_edge_size /= pixscale

        nimages = number_images_port(self.m_image_in_port)
        im_shape = self.m_image_in_port.get_shape()

        if self.m_norm:
            im_norm = np.linalg.norm(self.m_image_in_port.get_all(),
                                     ord="fro",
                                     axis=(1, 2))

        if self.m_resize is None:
            mask = create_mask((im_shape[-2], im_shape[-1]),
                               [self.m_cent_size, self.m_edge_size])

        else:
            im_res = np.zeros((nimages,
                               int(im_shape[-2]*self.m_resize),
                               int(im_shape[-1]*self.m_resize)))

            mask = create_mask((im_res.shape[-2], im_res.shape[-1]),
                               [self.m_cent_size, self.m_edge_size])

        for i in range(nimages):
            progress(i, nimages, "Running PSFpreparationModule...")

            image = self.m_image_in_port[i, ]

            if self.m_norm:
                # Normalize with the Frobenius norma
                image /= im_norm[i]

            if self.m_resize is not None:
                # Resample the data with a spline interpolation of the 5th order
                image = ndimage.interpolation.zoom(image,
                                                   zoom=[self.m_resize, self.m_resize],
                                                   order=5)

            if i == 0:
                if nimages == 1:
                    self.m_image_out_port.set_all(image*mask, data_dim=2)
                else:
                    self.m_image_out_port.set_all(image*mask, data_dim=3)
            else:
                self.m_image_out_port.append(image*mask, data_dim=3)

        if self.m_mask_out_port is not None:
            self.m_mask_out_port.set_all(mask)
            self.m_mask_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        if self.m_norm:
            self.m_image_out_port.add_attribute("norm", im_norm, static=False)

        if self.m_resize is not None:
            self.m_image_out_port.add_attribute("resize", self.m_resize, static=True)
            self.m_image_out_port.add_attribute("PIXSCALE", pixscale/self.m_resize)

        if self.m_cent_size is not None:
            self.m_image_out_port.add_attribute("cent_size",
                                                self.m_cent_size*pixscale,
                                                static=True)

        if self.m_edge_size is not None:
            self.m_image_out_port.add_attribute("edge_size",
                                                self.m_edge_size*pixscale,
                                                static=True)

        sys.stdout.write("Running PSFpreparationModule... [DONE]\n")
        sys.stdout.flush()


class AngleInterpolationModule(ProcessingModule):
    """
    Module for calculating the parallactic angle values by interpolating between the begin and end
    value of a data cube.
    """

    def __init__(self,
                 name_in="angle_interpolation",
                 data_tag="im_arr"):
        """
        Constructor of AngleInterpolationModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param data_tag: Tag of the database entry for which the parallactic angles are written as
                         attributes.
        :type data_tag: str

        :return: None
        """

        super(AngleInterpolationModule, self).__init__(name_in)

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

    def run(self):
        """
        Run method of the module. Calculates the parallactic angles of each frame by linearly
        interpolating between the start and end values of the data cubes. The values are written
        as attributes to *data_tag*. A correction of 360 deg is applied when the start and end
        values of the angles change sign at +/-180 deg.

        :return: None
        """

        parang_start = self.m_data_in_port.get_attribute("PARANG_START")
        parang_end = self.m_data_in_port.get_attribute("PARANG_END")

        steps = self.m_data_in_port.get_attribute("NFRAMES")
        ndit = self.m_data_in_port.get_attribute("NDIT")

        if not np.all(ndit == steps):
            warnings.warn("There is a mismatch between the NDIT and NFRAMES values. The "
                          "derotation angles are calculated with a linear interpolation by using "
                          "NFRAMES steps. A frame selection should be applied after the "
                          "derotation angles are calculated.")

        new_angles = []

        for i, _ in enumerate(parang_start):
            progress(i, len(parang_start), "Running AngleInterpolationModule...")

            if parang_start[i] < -170. and parang_end[i] > 170.:
                parang_start[i] += 360.

            elif parang_end[i] < -170. and parang_start[i] > 170.:
                parang_end[i] += 360.

            new_angles = np.append(new_angles,
                                   np.linspace(parang_start[i],
                                               parang_end[i],
                                               num=steps[i]))

        sys.stdout.write("Running AngleInterpolationModule... [DONE]\n")
        sys.stdout.flush()

        self.m_data_out_port.add_attribute("PARANG",
                                           new_angles,
                                           static=False)


class SortParangModule(ProcessingModule):
    """
    Module to sort the images and non-static attributes with increasing INDEX.
    """

    def __init__(self,
                 name_in="sort",
                 image_in_tag="im_arr",
                 image_out_tag="im_sort"):
        """
        Constructor of SortParangModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry with images that is written as output.
                              Should be different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(SortParangModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Sorts the images and relevant non-static attributes.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        memory = self._m_config_port.get_attribute("MEMORY")
        index = self.m_image_in_port.get_attribute("INDEX")

        index_new = np.zeros(index.shape, dtype=np.int)

        if "PARANG" in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute("PARANG")
            parang_new = np.zeros(parang.shape)

        else:
            parang_new = None

        if "STAR_POSITION" in self.m_image_in_port.get_all_non_static_attributes():
            star = self.m_image_in_port.get_attribute("STAR_POSITION")
            star_new = np.zeros(star.shape)

        else:
            star_new = None

        index_sort = np.argsort(index)

        nimages = self.m_image_in_port.get_shape()[0]

        frames = memory_frames(memory, nimages)

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running SortParangModule...")

            index_new[frames[i]:frames[i+1]] = index[index_sort[frames[i]:frames[i+1]]]

            if parang_new is not None:
                parang_new[frames[i]:frames[i+1]] = parang[index_sort[frames[i]:frames[i+1]]]

            if star_new is not None:
                star_new[frames[i]:frames[i+1]] = star[index_sort[frames[i]:frames[i+1]]]

            # h5py indexing elements must be in increasing order
            for _, item in enumerate(index_sort[frames[i]:frames[i+1]]):
                self.m_image_out_port.append(self.m_image_in_port[item, ], data_dim=3)

        sys.stdout.write("Running SortParangModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_attribute("INDEX", index_new, static=False)

        if parang_new is not None:
            self.m_image_out_port.add_attribute("PARANG", parang_new, static=False)

        if star_new is not None:
            self.m_image_out_port.add_attribute("STAR_POSITION", star_new, static=False)

        self.m_image_out_port.add_history_information("SortParangModule",
                                                      "images sorted by INDEX")

        self.m_image_out_port.close_port()


class AngleCalculationModule(ProcessingModule):
    """
    Module for calculating the parallactic angles. The start time of the observation is taken and
    multiples of the exposure time are added to derive the parallactic angle of each frame inside
    the cube. Instrument specific overheads are included. Written by Alexander Bohn (Leiden
    University).
    """

    def __init__(self,
                 instrument="NACO",
                 name_in="angle_calculation",
                 data_tag="im_arr"):
        """
        Constructor of AngleCalculationModule.

        :param instrument: Instrument name (*NACO*, *SPHERE/IRDIS*, or *SPHERE/IFS*)
        :type instrument: str
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param data_tag: Tag of the database entry for which the parallactic angles are written as
                         attributes.
        :type data_tag: str

        :return: None
        """

        super(AngleCalculationModule, self).__init__(name_in)

        # Parameters
        self.m_instrument = instrument

        # Set parameters according to choice of instrument
        if self.m_instrument == "NACO":

            # pupil offset in degrees
            self.m_pupil_offset = 0.            # No offset here

            # no overheads in cube mode, since cube is read out after all individual exposures
            # see NACO manual page 62 (v102)
            self.m_O_START = 0.
            self.m_DIT_DELAY = 0.
            self.m_ROT = 0.

            # rotator offset in degrees
            self.m_rot_offset = 89.44           # According to NACO manual page 65 (v102)

        elif self.m_instrument == "SPHERE/IRDIS":

            # pupil offset in degrees
            self.m_pupil_offset = -135.99       # According to SPHERE manual page 64 (v102)

            # overheads in cube mode (several NDITS) in hours
            self.m_O_START = 0.3 / 3600.        # According to SPHERE manual page 90/91 (v102)
            self.m_DIT_DELAY = 0.1 / 3600.      # According to SPHERE manual page 90/91 (v102)
            self.m_ROT = 0.838 / 3600.          # According to SPHERE manual page 90/91 (v102)

            # rotator offset in degrees
            self.m_rot_offset = 0.              # no offset here

        elif self.m_instrument == "SPHERE/IFS":

            # pupil offset in degrees
            self.m_pupil_offset = -135.99 - 100.48  # According to SPHERE manual page 64 (v102)

            # overheads in cube mode (several NDITS) in hours
            self.m_O_START = 0.3 / 3600.            # According to SPHERE manual page 90/91 (v102)
            self.m_DIT_DELAY = 0.2 / 3600.          # According to SPHERE manual page 90/91 (v102)
            self.m_ROT = 1.65 / 3600.               # According to SPHERE manual page 90/91 (v102)

            # rotator offset in degrees
            self.m_rot_offset = 0.                  # no offset here

        else:
            raise ValueError("The instrument argument should be set to either 'NACO', "
                             "'SPHERE/IRDIS', or 'SPHERE/IFS'.")

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

    def run(self):
        """
        Run method of the module. Calculates the parallactic angles from the position of the object
        on the sky and the telescope location on earth. The start of the observation is used to
        extrapolate for the observation time of each individual image of a data cube. The values
        are written as PARANG attributes to *data_tag*.

        :return: None
        """

        # Load cube sizes
        steps = self.m_data_in_port.get_attribute("NFRAMES")
        ndit = self.m_data_in_port.get_attribute("NDIT")

        if not np.all(ndit == steps):
            warnings.warn("There is a mismatch between the NDIT and NFRAMES values. A frame "
                          "selection should be applied after the derotation angles are "
                          "calculated.")

        if self.m_instrument == "SPHERE/IFS":
            warnings.warn("AngleCalculationModule has not been tested for SPHERE/IFS data.")

        # Load exposure time [hours]
        exptime = self.m_data_in_port.get_attribute("DIT")/3600.

        # Load telescope location
        tel_lat = self.m_data_in_port.get_attribute("LATITUDE")
        tel_lon = self.m_data_in_port.get_attribute("LONGITUDE")

        # Load target position [deg]
        ra = self.m_data_in_port.get_attribute("RA")
        dec = self.m_data_in_port.get_attribute("DEC")

        ra = np.mean(ra)
        dec = np.mean(dec)

        # Load start times of exposures
        obs_dates = self.m_data_in_port.get_attribute("DATE")

        # Load pupil positions during observations
        if self.m_instrument == "NACO":
            pupil_pos = self.m_data_in_port.get_attribute("PUPIL")

        elif self.m_instrument == "SPHERE/IRDIS":
            pupil_pos = np.zeros(steps.shape)

        elif self.m_instrument == "SPHERE/IFS":
            pupil_pos = np.zeros(steps.shape)

        new_angles = np.array([])
        pupil_pos_arr = np.array([])

        # Calculate parallactic angles for each cube
        for i, tmp_steps in enumerate(steps):

            # Create an ephem observer class to calculate local sidereal time
            obs = ephem.Observer()

            obs.lat = ephem.degrees(str(tel_lat))
            obs.long = ephem.degrees(str(tel_lon))

            obs.date = str(obs_dates[i].replace(b'T', b' ').decode("utf-8"))

            # Get sideral time in hours
            sid_time = str(obs.sidereal_time())

            # Get hours minutes and seconds
            h, m, s = sid_time.split(":")

            sid_time = (float(h) + (float(m) / 60.) + (float(s) / 3600.))

            # Extrapolate sideral times from start time of the cube for each frame of it
            sid_time_arr = np.linspace(sid_time+self.m_O_START,
                                       (sid_time+self.m_O_START) + (exptime+self.m_DIT_DELAY+ \
                                                 self.m_ROT)*(tmp_steps-1),
                                       tmp_steps)

            # Convert to degrees
            sid_time_arr_deg = sid_time_arr * 15.

            # Calculate hour angle in degrees
            hour_angle = sid_time_arr_deg - ra

            # Conversion to radians:
            hour_angle_rad = np.deg2rad(hour_angle)
            dec_rad = np.deg2rad(dec)
            lat_rad = np.deg2rad(tel_lat)

            p_angle = np.arctan2(np.sin(hour_angle_rad),
                                 (np.cos(dec_rad)*np.tan(lat_rad) - \
                                  np.sin(dec_rad)*np.cos(hour_angle_rad)))

            new_angles = np.append(new_angles, np.rad2deg(p_angle))
            pupil_pos_arr = np.append(pupil_pos_arr, np.ones(tmp_steps)*pupil_pos[i])

        # Correct for rotator (SPHERE) or pupil offset (NACO)
        if self.m_instrument == "NACO":
            # See NACO manual page 65 (v102)
            new_angles_corr = new_angles - (90. + (self.m_rot_offset-pupil_pos_arr))

        elif self.m_instrument == "SPHERE/IRDIS":
            # See SPHERE manual page 64 (v102)
            new_angles_corr = new_angles - self.m_pupil_offset

        elif self.m_instrument == "SPHERE/IFS":
            # See SPHERE manual page 64 (v102)
            new_angles_corr = new_angles - self.m_pupil_offset

        self.m_data_out_port.add_attribute("PARANG", new_angles_corr, static=False)

        sys.stdout.write("Running AngleCalculationModule... [DONE]\n")
        sys.stdout.flush()


class SDIpreparationModule(ProcessingModule):
    """
    Module for preparing continuum frames for SDI subtraction.
    """

    def __init__(self,
                 wavelength,
                 width,
                 name_in="SDI_preparation",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_SDI"):
        """
        Constructor of SDIpreparationModule.

        :param wavelength: Tuple with the central wavelengths of the line and continuum filter,
                           (line, continuum), in arbitrary but identical units.
        :type wavelength: (float, float)
        :param width: Tuple with the equivalent widths of the line and continuum filter,
                      (line, continuum), in arbitrary but identical units.
        :type width: (float, float)
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(SDIpreparationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_line_wvl = wavelength[0]
        self.m_cnt_wvl = wavelength[1]

        self.m_line_width = width[0]
        self.m_cnt_width = width[1]

    def run(self):
        """
        Run method of the module. Normalizes the images for the different filter widths,
        upscales the images, and crops the images to the initial image shape in order to
        align the PSF patterns.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        wvl_factor = self.m_line_wvl/self.m_cnt_wvl
        width_factor = self.m_line_width/self.m_cnt_width

        nimages = number_images_port(self.m_image_in_port)

        for i in range(nimages):
            progress(i, nimages, "Running SDIpreparationModule...")

            if nimages == 1:
                image = self.m_image_in_port.get_all()

            else:
                image = self.m_image_in_port[i, ]

            im_scale = width_factor * scale_image(image, wvl_factor, wvl_factor)

            if i == 0:
                npix_del = im_scale.shape[-1] - image.shape[-1]

                if npix_del%2 == 0:
                    npix_del_a = int(npix_del/2)
                    npix_del_b = int(npix_del/2)

                else:
                    npix_del_a = int((npix_del-1)/2)
                    npix_del_b = int((npix_del+1)/2)

            im_crop = im_scale[npix_del_a:-npix_del_b, npix_del_a:-npix_del_b]

            if npix_del%2 == 1:
                im_crop = shift_image(im_crop, (-0.5, -0.5), interpolation="spline")

            if nimages == 1:
                self.m_image_out_port.set_all(im_crop)

            else:
                self.m_image_out_port.append(im_crop, data_dim=3)

        sys.stdout.write("Running SDIpreparationModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        history = "(line, continuum) = ("+str(self.m_line_wvl)+", "+str(self.m_cnt_wvl)+")"
        self.m_image_out_port.add_history_information("Wavelength center", history)

        history = "(line, continuum) = ("+str(self.m_line_width)+", "+str(self.m_cnt_width)+")"
        self.m_image_out_port.add_history_information("Wavelength width", history)

        self.m_image_in_port.close_port()
