"""
Module for reading FITS files.
"""

from __future__ import absolute_import

import os
import sys
import warnings

import six
import numpy as np

from astropy.io import fits

from pynpoint.core.attributes import get_attributes
from pynpoint.core.processing import ReadingModule
from pynpoint.util.module import progress


class FitsReadingModule(ReadingModule):
    """
    Reads FITS files from the given *input_dir* or the default directory of the Pypeline. The FITS
    files need to contain either single images (2D) or cubes of images (3D). Individual images
    should have the same shape and type. The header of the FITS is scanned for the required static
    attributes (should be identical for each FITS file) and non-static attributes. Static entries
    will be saved as HDF5 attributes while non-static attributes will be saved as separate data
    sets in a subfolder of the database named *header_* + image_tag. If the FITS files in the input
    directory have changing static attributes or the shape of the input images is changing a
    warning appears. FitsReadingModule overwrites by default all existing data with the same tags
    in the central database.
    """

    def __init__(self,
                 name_in=None,
                 input_dir=None,
                 image_tag="im_arr",
                 overwrite=True,
                 check=True,
                 filenames=None):
        """
        Constructor of FitsReadingModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        input_dir : str
            Input directory where the FITS files are located. If not specified the Pypeline default
            directory is used.
        image_tag : str
            Tag of the read data in the HDF5 database. Non static header information is stored with
            the tag: *header_* + image_tag / header_entry_name.
        overwrite : bool
            Overwrite existing data and header in the central database.
        check : bool
            Check all the listed non-static attributes or ignore the attributes that are not always
            required (e.g. PARANG_START, DITHER_X).
        filenames : str or list(str, )
            If a string, then a path of a text file should be provided. This text file should
            contain a list of FITS files. If a list, then the paths of the FITS files should be
            provided directly. If set to None, the FITS files in the `input_dir` are read. All
            paths should be provided either relative to the Python working folder (i.e., the folder
            where Python is executed) or as absolute paths.

        Returns
        -------
        NoneType
            None
        """

        super(FitsReadingModule, self).__init__(name_in, input_dir)

        self.m_image_out_port = self.add_output_port(image_tag)

        self.m_overwrite = overwrite
        self.m_check = check
        self.m_filenames = filenames

        self.m_static = []
        self.m_non_static = []

        self.m_attributes = get_attributes()

        for key, value in six.iteritems(self.m_attributes):
            if value["config"] == "header" and value["attribute"] == "static":
                self.m_static.append(key)

        for key, value in six.iteritems(self.m_attributes):
            if value["attribute"] == "non-static":
                self.m_non_static.append(key)

        self.m_count = 0

        if not isinstance(filenames, (type(None), list, tuple, str)):
            raise TypeError("The 'filenames' parameter should contain a string or list with "
                            "strings.")

    def _read_single_file(self,
                          fits_file,
                          location,
                          overwrite_tags):
        """
        Internal function which reads a single FITS file and appends it to the database. The
        function gets a list of *overwriting_tags*. If a new key (header entry or image data) is
        found that is not on this list the old entry is overwritten if *self.m_overwrite* is
        active. After replacing the old entry the key is added to the *overwriting_tags*. This
        procedure guaranties that all previous database information, that does not belong to the
        new data set that is read by FitsReadingModule is replaced and the rest is kept.

        Parameters
        ----------
        fits_file : str
            Name of the FITS file.
        location : str
            Directory where the FITS file is located.
        overwrite_tags : bool
            The list of database tags that will not be overwritten.

        Returns
        -------
        astropy.io.fits.header.Header
            FITS header.
        tuple(int, )
            Image shape.
        """

        hdulist = fits.open(os.path.join(location, fits_file))
        images = hdulist[0].data.byteswap().newbyteorder()

        if self.m_overwrite and self.m_image_out_port.tag not in overwrite_tags:
            overwrite_tags.append(self.m_image_out_port.tag)

            self.m_image_out_port.set_all(images, data_dim=3)
            self.m_image_out_port.del_all_attributes()

        else:
            self.m_image_out_port.append(images, data_dim=3)

        header = hdulist[0].header

        fits_header = []
        for key in header:
            fits_header.append(str(key)+" = "+str(header[key]))

        hdulist.close()

        header_out_port = self.add_output_port('fits_header/'+fits_file)
        header_out_port.set_all(fits_header)

        return header, images.shape

    def _txt_file_list(self):
        """
        Internal function to import a list of FITS files from a text file.
        """

        with open(self.m_filenames) as file_obj:
            files = file_obj.readlines()

        files = [x.strip() for x in files] # get rid of newlines

        return list(filter(None, files)) # get rid of empty lines

    def _static_attributes(self,
                           fits_file,
                           header):
        """
        Internal function which adds the static attributes to the central database.

        Parameters
        ----------
        fits_file : str
            Name of the FITS file.
        header : astropy.io.fits.header.Header
            Header information from the FITS file that is read.

        Returns
        -------
        NoneType
            None
        """

        for item in self.m_static:

            if self.m_check:
                fitskey = self._m_config_port.get_attribute(item)

                if isinstance(fitskey, np.bytes_):
                    fitskey = str(fitskey.decode("utf-8"))

                if fitskey != "None":
                    if fitskey in header:
                        status = self.m_image_out_port.check_static_attribute(item,
                                                                              header[fitskey])

                        if status == 1:
                            self.m_image_out_port.add_attribute(item,
                                                                header[fitskey],
                                                                static=True)

                        if status == -1:
                            warnings.warn("Static attribute %s has changed. Possibly the current "
                                          "file %s does not belong to the data set '%s'. Attribute "
                                          "value is updated." \
                                          % (fitskey, fits_file, self.m_image_out_port.tag))

                        elif status == 0:
                            pass

                    else:
                        warnings.warn("Static attribute %s (=%s) not found in the FITS header." \
                                      % (item, fitskey))

    def _non_static_attributes(self,
                               header):
        """
        Internal function which adds the non-static attributes to the central database.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            Header information from the FITS file that is read.

        Returns
        -------
        NoneType
            None
        """

        for item in self.m_non_static:
            if self.m_check:
                if self.m_attributes[item]["config"] == "header":
                    fitskey = self._m_config_port.get_attribute(item)

                    # if type(fitskey) == np.bytes_:
                    #     fitskey = str(fitskey.decode("utf-8"))

                    if fitskey != "None":
                        if fitskey in header:
                            self.m_image_out_port.append_attribute_data(item, header[fitskey])

                        elif header['NAXIS'] == 2 and item == 'NFRAMES':
                            self.m_image_out_port.append_attribute_data(item, 1)

                        else:
                            warnings.warn("Non-static attribute %s (=%s) not found in the "
                                          "FITS header." % (item, fitskey))

                            self.m_image_out_port.append_attribute_data(item, -1)

    def _extra_attributes(self,
                          fits_file,
                          location,
                          shape):
        """
        Internal function which adds extra attributes to the central database.

        Parameters
        ----------
        fits_file : str
            Name of the FITS file.
        location : str
            Directory where the FITS file is located.
        shape : tuple(int, )
            Shape of the images.

        Returns
        -------
        NoneType
            None
        """

        pixscale = self._m_config_port.get_attribute('PIXSCALE')

        if len(shape) == 2:
            nimages = 1
        elif len(shape) == 3:
            nimages = shape[0]

        index = np.arange(self.m_count, self.m_count+nimages, 1)

        for _, item in enumerate(index):
            self.m_image_out_port.append_attribute_data("INDEX", item)

        self.m_image_out_port.append_attribute_data("FILES", os.path.join(location, fits_file))
        self.m_image_out_port.add_attribute("PIXSCALE", pixscale, static=True)

        self.m_count += nimages

    def run(self):
        """
        Run method of the module. Looks for all FITS files in the input directory and imports the
        images into the database. Note that previous database information is overwritten if
        ``overwrite=True``. The filenames are stored as attributes.

        Returns
        -------
        NoneType
            None
        """

        files = []

        if isinstance(self.m_filenames, str):
            files = self._txt_file_list()
            location = os.getcwd()

        elif isinstance(self.m_filenames, (list, tuple)):
            files = self.m_filenames
            location = os.getcwd()

        elif isinstance(self.m_filenames, type(None)):
            location = os.path.join(self.m_input_location, '')

            for filename in os.listdir(location):
                if filename.endswith('.fits') and not filename.startswith('._'):
                    files.append(filename)

            assert(files), 'No FITS files found in %s.' % self.m_input_location

        files.sort()

        overwrite_tags = []

        for i, fits_file in enumerate(files):
            progress(i, len(files), "Running FitsReadingModule...")

            header, shape = self._read_single_file(fits_file, location, overwrite_tags)

            self._static_attributes(fits_file, header)
            self._non_static_attributes(header)
            self._extra_attributes(fits_file, location, shape)

            self.m_image_out_port.flush()

        sys.stdout.write("Running FitsReadingModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.close_port()
