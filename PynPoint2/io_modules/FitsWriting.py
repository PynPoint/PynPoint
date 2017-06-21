"""
Module for writing data as .fits file.
"""
# external modules
import os

from astropy.io import fits

from PynPoint2.core.Processing import WritingModule

import warnings


class WriteAsSingleFitsFile(WritingModule):
    """
    Module for writing a data set of the central .hdf5 data base as .fits file. The data and all
    attached attributes will be saved. Beside typical image stacks it is possible to export for
    example non static header information. To choose the data set from the data base its tag
    / key has to be specified. WriteAsSingleFitsFile is a Writing Module and supports to use the
    Pypeline default output directory as well as a own location. (see
    :class:`PynPoint.core.Processing.WritingModule` for more information)
    """

    def __init__(self,
                 file_name,
                 name_in="fits_writing",
                 output_dir=None,
                 data_tag="im_arr"):
        """
        Constructor of the WriteAsSingleFitsFile module. It needs the name of the output file as
        well as the dataset tag which has to exported into that file. See class documentation for
        more information.

        :param name_in: Name of the module instance. Used as unique identifier in the Pypeline
                        dictionary. (See :class:`PynPoint.core.Pypeline.Pypeline` for more
                        information)
        :type name_in: str
        :param file_name: Name of the .fits output file. Needs to end with *.fits*
        :type file_name: str
        :param output_dir: Output directory where the .fits file will be stored. If no folder is
                           specified the Pypeline default is chosen.
        :type output_dir: str
        :param data_tag: Tag of the data base entry the module has to export as .fits file.
        :type data_tag: str
        :return: None
        """
        super(WriteAsSingleFitsFile, self).__init__(name_in=name_in,
                                                    output_dir=output_dir)
        if not isinstance(file_name, str):
            raise ValueError("Output file_name needs to be a String")

        if not file_name.endswith(".fits"):
            raise ValueError("Output file_name needs to end with .fits")

        self.m_file_name = file_name
        self.m_data_port = self.add_input_port(data_tag)

    def run(self):
        """
        Run method of the module. Creates a .fits file and saves the data as well as its attributes.

        :return: None
        """

        if not self.m_output_location.endswith('/'):
            out_name = self.m_output_location + '/' + self.m_file_name
        else:
            out_name = self.m_output_location + self.m_file_name

        # remove old file if exists
        if os.path.isfile(out_name):
            warnings.warn('Overwriting old file %s' % out_name)
            os.remove(out_name)

        # Attributes
        prihdr = fits.Header()
        attributes = self.m_data_port.get_all_static_attributes()
        for attr in attributes:
            if len(attr) > 8:
                prihdr["hierarch " + attr] = attributes[attr]
            else:
                prihdr[attr] = attributes[attr]

        # Data
        hdu = fits.PrimaryHDU(self.m_data_port.get_all(),
                              header=prihdr)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(out_name)

        self.m_data_port.close_port()
