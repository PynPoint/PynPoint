"""
Modules for writing data as text file.
"""

from __future__ import absolute_import

import os
import sys

import numpy as np

from pynpoint.core.processing import WritingModule


class TextWritingModule(WritingModule):
    """
    Module for writing a 1D or 2D data set from the central HDF5 database as text file.
    TextWritingModule is a :class:`pynpoint.core.processing.WritingModule` and supports
    the use of the Pypeline default output directory as well as a specified location.
    """

    def __init__(self,
                 file_name,
                 name_in="text_writing",
                 output_dir=None,
                 data_tag="im_arr",
                 header=None):
        """
        Constructor of TextWritingModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        file_name : str
            Name of the output file.
        output_dir : str
            Output directory where the text file will be stored. If no path is specified then the
            Pypeline default output location is used.
        data_tag : str
            Tag of the database entry from which data is exported.
        header : str
            Header that is written at the top of the text file.

        Returns
        -------
        NoneType
            None
        """

        super(TextWritingModule, self).__init__(name_in, output_dir)

        if not isinstance(file_name, str):
            raise ValueError("Output 'file_name' needs to be a string.")

        self.m_data_port = self.add_input_port(data_tag)

        self.m_file_name = file_name
        self.m_header = header

    def run(self):
        """
        Run method of the module. Saves the specified data from the database to a text file.

        Returns
        -------
        NoneType
            None
        """

        if self.m_header is None:
            self.m_header = ""

        sys.stdout.write("Running TextWritingModule...")
        sys.stdout.flush()

        out_name = os.path.join(self.m_output_location, self.m_file_name)

        data = self.m_data_port.get_all()

        if data.ndim == 3 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)

        if data.ndim > 2:
            raise ValueError("Only 1D or 2D arrays can be written to a text file.")

        if data.dtype == "int32" or data.dtype == "int64":
            np.savetxt(out_name, data, header=self.m_header, comments='# ', fmt="%i")

        elif data.dtype == "float32" or data.dtype == "float64":
            np.savetxt(out_name, data, header=self.m_header, comments='# ')

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_data_port.close_port()


class ParangWritingModule(WritingModule):
    """
    Module for writing a list of parallactic angles to a text file.
    """

    def __init__(self,
                 file_name="parang.dat",
                 name_in="parang_writing",
                 output_dir=None,
                 data_tag="im_arr",
                 header="Parallactic angle [deg]"):
        """
        Constructor of ParangWritingModule.

        Parameters
        ----------
        file_name : str
            Name of the output file.
        name_in : str
            Unique name of the module instance.
        output_dir : str
            Output directory where the text file will be stored. If no path is specified then the
            Pypeline default output location is used.
        data_tag : str
            Tag of the database entry from which the PARANG attribute is read.
        header : str
            Header that is written at the top of the text file.

        Returns
        -------
        NoneType
            None
        """

        super(ParangWritingModule, self).__init__(name_in, output_dir)

        if not isinstance(file_name, str):
            raise ValueError("Output 'file_name' needs to be a string.")

        self.m_data_port = self.add_input_port(data_tag)

        self.m_file_name = file_name
        self.m_header = header

    def run(self):
        """
        Run method of the module. Writes the parallactic angles from the PARANG attribute of
        the specified database tag to a a text file.

        Returns
        -------
        NoneType
            None
        """

        sys.stdout.write("Running ParangWritingModule...")
        sys.stdout.flush()

        if self.m_header is None:
            self.m_header = ""

        out_name = os.path.join(self.m_output_location, self.m_file_name)

        if "PARANG" not in self.m_data_port.get_all_non_static_attributes():
            raise ValueError("The PARANG attribute is not present in '%s'." % self.m_data_port.tag)

        parang = self.m_data_port.get_attribute("PARANG")

        np.savetxt(out_name, parang, header=self.m_header, comments='# ')

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_data_port.close_port()


class AttributeWritingModule(WritingModule):
    """
    Module for writing a 1D or 2D array of non-static attributes to a text file.
    """

    def __init__(self,
                 file_name="attributes.dat",
                 name_in="attribute_writing",
                 output_dir=None,
                 data_tag="im_arr",
                 attribute="INDEX",
                 header=None):
        """
        Constructor of AttributeWritingModule.

        Parameters
        ----------
        file_name : str
            Name of the output file.
        name_in : str
            Unique name of the module instance.
        output_dir : str
            Output directory where the text file will be stored. If no path is specified then the
            Pypeline default output location is used.
        data_tag : str
            Tag of the database entry from which the PARANG attribute is read.
        attribute : str
            Name of the non-static attribute as given in the central database (e.g., "INDEX" or
            "STAR_POSITION").
        header : str
            Header that is written at the top of the text file.

        Returns
        -------
        NoneType
            None
        """

        super(AttributeWritingModule, self).__init__(name_in, output_dir)

        if not isinstance(file_name, str):
            raise ValueError("Output 'file_name' needs to be a string.")

        self.m_data_port = self.add_input_port(data_tag)

        self.m_file_name = file_name
        self.m_attribute = attribute
        self.m_header = header

    def run(self):
        """
        Run method of the module. Writes the non-static attributes (1D or 2D) to a a text file.

        Returns
        -------
        NoneType
            None
        """

        if self.m_header is None:
            self.m_header = ""

        sys.stdout.write("Running AttributeWritingModule...")
        sys.stdout.flush()

        out_name = os.path.join(self.m_output_location, self.m_file_name)

        if self.m_attribute not in self.m_data_port.get_all_non_static_attributes():
            raise ValueError("The '%s' attribute is not present in '%s'."
                             % (self.m_attribute, self.m_data_port.tag))

        values = self.m_data_port.get_attribute(self.m_attribute)

        np.savetxt(out_name, values, header=self.m_header, comments='# ')

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_data_port.close_port()
