"""
Module for writing data as FITS file.
"""

from __future__ import absolute_import

import os
import sys
import warnings

from astropy.io import fits

from PynPoint.Core.Processing import WritingModule


class FitsWritingModule(WritingModule):
    """
    Module for writing a data set of the central HDF5 database as FITS file. The data and all
    attached attributes will be saved. Besides typical image stacks it is possible to export for
    example non-static header information. To choose the data set from the database its tag
    / key has to be specified. FitsWritingModule is a Writing Module and supports to use the
    Pypeline default output directory as well as a own location. See
    :class:`PynPoint.core.Processing.WritingModule` for more information. Note that per default
    this module will overwrite an existing FITS file with the same filename.
    """

    def __init__(self,
                 file_name,
                 name_in="fits_writing",
                 output_dir=None,
                 data_tag="im_arr",
                 data_range=None,
                 overwrite=True):
        """
        Constructor of FitsWritingModule. It needs the name of the output file as well as
        the dataset tag which has to exported into that file. See class documentation for more
        information.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param file_name: Name of the FITS output file. Requires the FITS extension.
        :type file_name: str
        :param output_dir: Output directory where the FITS file will be stored. If no folder is
                           specified the Pypeline default is chosen.
        :type output_dir: str
        :param data_tag: Tag of the database entry the module has to export as FITS file.
        :type data_tag: str
        :param data_range: A two element tuple which specifies a begin and end frame of the export.
                           This can be used to save a subsets of huge dataset. If None the whole
                           dataset will be exported.
        :type data_range: tuple
        :param overwrite: Overwrite existing FITS file with identical filename.
        :type overwrite: bool

        :return: None
        """

        super(FitsWritingModule, self).__init__(name_in=name_in, output_dir=output_dir)

        if not isinstance(file_name, str):
            raise ValueError("Output 'file_name' needs to be a string.")

        if not file_name.endswith(".fits"):
            raise ValueError("Output 'file_name' requires the FITS extension.")

        self.m_file_name = file_name
        self.m_data_port = self.add_input_port(data_tag)
        self.m_range = data_range
        self.m_overwrite = overwrite

    def run(self):
        """
        Run method of the module. Creates a FITS file and saves the data as well as the
        corresponding attributes.

        :return: None
        """

        out_name = os.path.join(self.m_output_location, self.m_file_name)

        sys.stdout.write("Running FitsWritingModule...")
        sys.stdout.flush()

        if os.path.isfile(out_name) and not self.m_overwrite:
            warnings.warn("Filename already present. Use overwrite=True to overwrite an existing "
                          "FITS file.")

        else:
            prihdr = fits.Header()
            attributes = self.m_data_port.get_all_static_attributes()

            for attr in attributes:
                if len(attr) > 8:
                    prihdr["hierarch "+attr] = attributes[attr]
                else:
                    prihdr[attr] = attributes[attr]

            if self.m_range is None:
                hdu = fits.PrimaryHDU(self.m_data_port.get_all(),
                                      header=prihdr)
            else:
                hdu = fits.PrimaryHDU(self.m_data_port[self.m_range[0]:self.m_range[1], ],
                                      header=prihdr)

            hdulist = fits.HDUList([hdu])
            hdulist.writeto(out_name, overwrite=self.m_overwrite)

            sys.stdout.write(" [DONE]\n")
            sys.stdout.flush()

        self.m_data_port.close_port()
