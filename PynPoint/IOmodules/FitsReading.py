"""
Module for reading FITS files.
"""

import os
import sys
import warnings

import numpy as np

from astropy.io import fits

from PynPoint.Core.Processing import ReadingModule
from PynPoint.Util.Progress import progress


class FitsReadingModule(ReadingModule):
    """
    Reads FITS files from the given *input_dir* or the default directory of the Pypeline. The FITS
    files need to contain either single images (2D) or cubes of images (3D). Individual images
    should have the same shape and type. The header of the FITS is scanned for the required static
    attributes (should be identical for a FITS files) and non-static attributes. Static entries
    will be saved as HDF5 attributes while non-static attributes will be saved as separate data
    sets in a subfolder of the database named *header_ + image_tag*. If the FITS files in the input
    directory have changing static attributes or the shape of the input images is changing a
    warning is caused. FitsReadingModule overwrites by default all existing data with the same tags
    in the Pynpoint database. Note that PynPoint only supports the processing of square images so
    rectangular images should be made square with e.g. CropImagesModule or RemoveLinesModule.
    """

    def __init__(self,
                 name_in=None,
                 input_dir=None,
                 image_tag="im_arr",
                 overwrite=True,
                 check=True):
        """
        Constructor of FitsReadingModule. As all Reading Modules it has a name and
        input directory (:class:`PynPoint.core.Processing.ReadingModule`). In addition a image_tag
        can be chosen which will be the tag / key of the read data in the .hdf5 database.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param input_dir: Input directory where the FITS files are located. If not specified the
                          Pypeline default directory is used.
        :type input_dir: str
        :param image_tag: Tag of the read data in the .hdf5 data base. Non static header
                          information is stored with the tag: *header_* + image_tag /
                          #header_entry_name#.
        :type image_tag: str
        :param overwrite: If True existing data (header and the actual data set) in the .hdf5
                          data base will be overwritten.
        :type overwrite: bool
        :param check: Check all the listed non-static attributes or leave out the attributes that
                      are not always required (e.g. PARANG_START, DITHER_X).
        :type check: bool

        :return: None
        """

        super(FitsReadingModule, self).__init__(name_in, input_dir)

        self.m_image_out_port = self.add_output_port(image_tag)

        self.m_image_tag = image_tag
        self.m_overwrite = overwrite

        self.m_static = ['INSTRUMENT']

        self.m_non_static = ['NFRAMES',
                             'EXP_NO',
                             'NDIT',
                             'PARANG_START',
                             'PARANG_END',
                             'PARANG',
                             'DITHER_X',
                             'DITHER_Y']

        self.m_attr_check = np.ones(len(self.m_non_static), dtype=bool)

        if not check:
            self.m_attr_check[3:8] = False

        self.m_count = 0

    def _read_single_file(self,
                          fits_file,
                          location,
                          overwrite_keys):
        """
        Internal function which reads a single FITS file and appends it to the database. The
        function gets a list of *overwriting_keys*. If a new key (header entry or image data) is
        found that is not on this list the old entry is overwritten if *self.m_overwrite* is
        active. After replacing the old entry the key is added to the *overwriting_keys*. This
        procedure guaranties that all old data base information, that does not belong to the
        new data set that is read by FitsReadingModule is replaced and the rest is kept.

        :param fits_file: Name of the FITS file.
        :type fits_file: str
        :param location: Directory where the FITS file is located.
        :type location: str
        :param overwrite_keys: The list of keys that will not be overwritten by the function
        :type overwrite_keys: bool

        :return: None
        """

        pixscale = self._m_config_port.get_attribute('PIXSCALE')

        hdulist = fits.open(location + fits_file)
        images = hdulist[0].data.byteswap().newbyteorder()

        if self.m_overwrite and self.m_image_tag not in overwrite_keys:
            self.m_image_out_port.set_all(images, data_dim=3)
            self.m_image_out_port.del_all_attributes()
            overwrite_keys.append(self.m_image_tag)

        else:
            self.m_image_out_port.append(images, data_dim=3)

        if images.ndim == 3:
            nimages = images.shape[0]
        elif images.ndim == 2:
            nimages = 1

        # store header info
        tmp_header = hdulist[0].header

        # static attributes
        for item in self.m_static:
            fitskey = self._m_config_port.get_attribute(item)

            if fitskey != "None":

                if fitskey in tmp_header:
                    value = tmp_header[fitskey]
                    status = self.m_image_out_port.check_static_attribute(item, value)

                    if status == 1:
                        self.m_image_out_port.add_attribute(item, value, static=True)

                    if status == -1:
                        warnings.warn('Static attribute %s has changed. Probably the current '
                                      'file %s does not belong to the data set "%s" of the PynPoint'
                                      ' database. Updating attribute...' \
                                      % (fitskey, fits_file, self.m_image_tag))

                    elif status == 0:
                        # Attribute is known and is still the same
                        pass

                else:
                    warnings.warn("Static attribute %s (=%s) not found in the FITS header." \
                                  % (item, fitskey))

        # non-static attributes
        for i, item in enumerate(self.m_non_static):

            if self.m_attr_check[i]:

                if item == 'PARANG':
                    fitskey = 'PARANG'

                else:
                    fitskey = self._m_config_port.get_attribute(item)

                if fitskey != "None":

                    if fitskey in tmp_header:
                        value = tmp_header[fitskey]
                        self.m_image_out_port.append_attribute_data(item, value)

                    elif tmp_header['NAXIS'] == 2 and item == 'NFRAMES':
                        self.m_image_out_port.append_attribute_data(item, 1)

                    elif item == 'PARANG':
                        continue

                    else:
                        warnings.warn("Non-static attribute %s (=%s) not found in the FITS "
                                      "header." % (item, fitskey))
                        self.m_image_out_port.append_attribute_data(item, -1)

        fits_header = []
        for key in tmp_header:
            if key:
                fits_header.append(str(key)+" = "+str(tmp_header[key]))

        hdulist.close()

        header_out_port = self.add_output_port('fits_header/'+fits_file)
        header_out_port.set_all(fits_header)

        index = np.arange(self.m_count, self.m_count+nimages, 1)

        for i, item in enumerate(index):
            self.m_image_out_port.append_attribute_data("INDEX", item)

        self.m_image_out_port.append_attribute_data("FILES", location+fits_file)
        self.m_image_out_port.add_attribute("PIXSCALE", pixscale, static=True)
        self.m_image_out_port.flush()

        self.m_count += nimages

    def run(self):
        """
        Run method of the module. Looks for all FITS files in the input directory and reads them
        using the internal function _read_single_file(). Note that if *overwrite* is True old
        database information is overwritten. The module saves the number of files and filenames as
        attributes. Note that PynPoint only supports the processing of square images so, in case
        required, images should be made square (e.g. with CropImagesModule or RemoveLinesModule)
        after importing the FITS files.

        :return: None
        """

        location = os.path.join(self.m_input_location, '')

        # search for fits files
        files = []
        for tmp_file in os.listdir(location):
            if tmp_file.endswith('.fits'):
                files.append(tmp_file)
        files.sort()

        assert(files), 'Error no FITS files found in %s' % self.m_input_location

        # overwrite_keys save the database keys which were updated. Used for overwriting only
        overwrite_keys = []

        # read file and append data to storage
        for i, fits_file in enumerate(files):
            progress(i, len(files), "Running FitsReadingModule...")

            self._read_single_file(fits_file, location, overwrite_keys)
            self.m_image_out_port.flush()

        sys.stdout.write("Running FitsReadingModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.close_database()
