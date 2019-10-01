"""
Module for reading FITS files.
"""

import os
import time

from typing import Union, Tuple, List

from astropy.io import fits
from typeguard import typechecked

from pynpoint.core.processing import ReadingModule
from pynpoint.util.attributes import set_static_attr, set_nonstatic_attr, set_extra_attr
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

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 input_dir: str = None,
                 image_tag: str = 'im_arr',
                 overwrite: bool = True,
                 check: bool = True,
                 filenames: Union[str, List[str]] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        input_dir : str, None
            Input directory where the FITS files are located. If not specified the Pypeline default
            directory is used.
        image_tag : str
            Tag of the read data in the HDF5 database. Non static header information is stored with
            the tag: *header_* + image_tag / header_entry_name.
        overwrite : bool
            Overwrite existing data and header in the central database.
        check : bool
            Print a warning if certain attributes from the configuration file are not present in
            the FITS header. If set to `False`, attributes are still written to the dataset but
            there will be no warning if a keyword is not found in the FITS header.
        filenames : str, list(str, ), None
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

    @typechecked
    def read_single_file(self,
                         fits_file: str,
                         overwrite_tags: list) -> Tuple[fits.header.Header, tuple]:
        """
        Function which reads a single FITS file and appends it to the database. The function gets
        a list of *overwriting_tags*. If a new key (header entry or image data) is found that is
        not on this list the old entry is overwritten if *self.m_overwrite* is active. After
        replacing the old entry the key is added to the *overwriting_tags*. This procedure
        guaranties that all previous database information, that does not belong to the new data
        set that is read by FitsReadingModule is replaced and the rest is kept.

        Parameters
        ----------
        fits_file : str
            Absolute path and filename of the FITS file.
        overwrite_tags : list(str, )
            The list of database tags that will not be overwritten.

        Returns
        -------
        astropy.io.fits.header.Header
            FITS header.
        tuple(int, )
            Image shape.
        """

        hdulist = fits.open(fits_file)
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
            fits_header.append(str(key)+' = '+str(header[key]))

        hdulist.close()

        header_out_port = self.add_output_port('fits_header/'+os.path.basename(fits_file))
        header_out_port.set_all(fits_header)

        return header, images.shape

    @typechecked
    def _txt_file_list(self) -> list:
        """
        Internal function to import a list of FITS files from a text file.
        """

        with open(self.m_filenames) as file_obj:
            files = file_obj.readlines()

        # remove newlines
        files = [x.strip() for x in files]

        # remove of empty lines
        files = filter(None, files)

        return list(files)

    @typechecked
    def run(self) -> None:
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

            for item in files:
                if not os.path.isfile(item):
                    raise ValueError(f'The file {item} does not exist. Please check that the '
                                     f'path is correct.')

        elif isinstance(self.m_filenames, list):
            files = self.m_filenames

            for item in files:
                if not os.path.isfile(item):
                    raise ValueError(f'The file {item} does not exist. Please check that the '
                                     f'path is correct.')

        elif isinstance(self.m_filenames, type(None)):
            for filename in os.listdir(self.m_input_location):
                if filename.endswith('.fits') and not filename.startswith('._'):
                    files.append(os.path.join(self.m_input_location, filename))

            assert(files), 'No FITS files found in %s.' % self.m_input_location

        files.sort()

        overwrite_tags = []
        first_index = 0

        start_time = time.time()
        for i, fits_file in enumerate(files):
            progress(i, len(files), 'Reading FITS files...', start_time)

            header, shape = self.read_single_file(fits_file, overwrite_tags)

            if len(shape) == 2:
                nimages = 1

            elif len(shape) == 3:
                nimages = shape[0]

            set_static_attr(fits_file=fits_file,
                            header=header,
                            config_port=self._m_config_port,
                            image_out_port=self.m_image_out_port,
                            check=self.m_check)

            set_nonstatic_attr(header=header,
                               config_port=self._m_config_port,
                               image_out_port=self.m_image_out_port,
                               check=self.m_check)

            set_extra_attr(fits_file=fits_file,
                           nimages=nimages,
                           config_port=self._m_config_port,
                           image_out_port=self.m_image_out_port,
                           first_index=first_index)

            first_index += nimages

            self.m_image_out_port.flush()

        self.m_image_out_port.close_port()
