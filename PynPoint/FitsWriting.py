"""
Module for writing data as .fits file. Note non static Attributes usually stored in the hdf5 header
folder can not be written.
"""
# external modules
import warnings
import os
from astropy.io import fits

# own classes
from Processing import WritingModule


class WriteAsSingleFitsFile(WritingModule):

    def __init__(self,
                 name_in,
                 file_name,
                 output_dir=None,
                 data_tag="im_arr"):
        super(WriteAsSingleFitsFile, self).__init__(name_in=name_in,
                                                    output_dir=output_dir)
        if type(file_name) is not str:
            raise ValueError("Output file_name needs to be a String")

        if not file_name.endswith(".fits"):
            raise ValueError("Output file_name needs to end with .fits")

        self.m_file_name = file_name
        self.m_data_tag = data_tag
        self.add_input_port(data_tag)

    def run(self):

        if not self.m_output_location.endswith('/'):
            out_name = self.m_output_location +'/' + self.m_file_name
        else:
            out_name = self.m_output_location + self.m_file_name

        # remove old file if exists
        if os.path.isfile(out_name):
            warnings.warn('Overwriting old file %s' %out_name)
            os.remove(out_name)

        # Attributes
        prihdr = fits.Header()
        attributes = self._m_input_ports[self.m_data_tag].get_all_attributes()
        for attr in attributes:
            if len(attr) > 8:
                prihdr["hierarch " + attr] = attributes[attr]
            else:
                prihdr[attr] = attributes[attr]

        # Data
        hdu = fits.PrimaryHDU(self._m_input_ports[self.m_data_tag].get_all(),
                              header=prihdr)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(out_name)

        self._m_input_ports[self.m_data_tag].close_port()
