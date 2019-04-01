"""
Modules for reading data from a text file.
"""

from __future__ import absolute_import

import os
import sys
import warnings

import numpy as np

from pynpoint.core.attributes import get_attributes
from pynpoint.core.processing import ReadingModule


class ParangReadingModule(ReadingModule):
    """
    Module for reading a list of parallactic angles from a text file.
    """

    def __init__(self,
                 file_name,
                 name_in="parang_reading",
                 input_dir=None,
                 data_tag="im_arr",
                 overwrite=False):
        """
        Constructor of ParangReadingModule.

        Parameters
        ----------
        file_name : str
            Name of the input file with a list of parallactic angles (deg). Should be equal in size
            to the number of images in *data_tag*.
        name_in : str
            Unique name of the module instance.
        input_dir : str
            Input directory where the text file is located. If not specified the Pypeline default
            directory is used.
        data_tag : str
            Tag of the database entry to which the PARANG attribute is written.
        overwrite : bool
            Overwrite if the PARANG attribute already exists.

        Returns
        -------
        NoneType
            None
        """
        super(ParangReadingModule, self).__init__(name_in, input_dir)

        if not isinstance(file_name, str):
            raise ValueError("Input 'file_name' needs to be a string.")

        self.m_data_port = self.add_output_port(data_tag)

        self.m_file_name = file_name
        self.m_overwrite = overwrite

    def run(self):
        """
        Run method of the module. Reads the parallactic angles from a text file and writes the
        values as non-static attribute (PARANG) to the database tag.

        Returns
        -------
        NoneType
            None
        """

        sys.stdout.write("Running ParangReadingModule...")
        sys.stdout.flush()

        parang = np.loadtxt(os.path.join(self.m_input_location, self.m_file_name))

        if parang.ndim != 1:
            raise ValueError("The input file %s should contain a 1D data set with the parallactic "
                             "angles." % self.m_file_name)

        status = self.m_data_port.check_non_static_attribute("PARANG", parang)

        if status == 1:
            self.m_data_port.add_attribute("PARANG", parang, static=False)

        elif status == -1 and self.m_overwrite:
            self.m_data_port.add_attribute("PARANG", parang, static=False)

        elif status == -1 and not self.m_overwrite:
            warnings.warn("The PARANG attribute is already present. Set the 'overwrite' parameter "
                          "to True in order to overwrite the values with "+str(self.m_file_name)+
                          ".")

        elif status == 0:
            warnings.warn("The PARANG attribute is already present and contains the same values "
                          "as are present in "+str(self.m_file_name)+".")

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_data_port.close_port()


class AttributeReadingModule(ReadingModule):
    """
    Module for reading a list of values from a text file and appending them as a non-static
    attributes to a dataset.
    """

    def __init__(self,
                 file_name,
                 attribute,
                 name_in="attribute_reading",
                 input_dir=None,
                 data_tag="im_arr",
                 overwrite=False):
        """
        Constructor of AttributeReadingModule.

        Parameters
        ----------
        file_name : str
            Name of the input file with a list of values.
        attribute : str
            Name of the attribute as to be written in the database.
        name_in : str
            Unique name of the module instance.
        input_dir : str
            Input directory where the text file is located. If not specified the Pypeline default
            directory is used.
        data_tag : str
            Tag of the database entry to which the attribute is written.
        overwrite : bool
            Overwrite if the attribute is already exists.

        Returns
        -------
        NoneType
            None
        """

        super(AttributeReadingModule, self).__init__(name_in, input_dir)

        if not isinstance(file_name, str):
            raise ValueError("Input 'file_name' needs to be a string.")

        self.m_data_port = self.add_output_port(data_tag)

        self.m_file_name = file_name
        self.m_attribute = attribute
        self.m_overwrite = overwrite

    def run(self):
        """
        Run method of the module. Reads a list of values from a text file and writes them as
        non-static attribute to a dataset.

        Returns
        -------
        NoneType
            None
        """

        sys.stdout.write("Running AttributeReadingModule...")
        sys.stdout.flush()

        attributes = get_attributes()

        if self.m_attribute not in attributes:
            raise ValueError("'%s' is not a valid attribute." % self.m_attribute)

        values = np.loadtxt(os.path.join(self.m_input_location, self.m_file_name),
                            dtype=attributes[self.m_attribute]["type"])

        if values.ndim != 1:
            raise ValueError("The input file %s should contain a 1D list with attributes."
                             % self.m_file_name)

        status = self.m_data_port.check_non_static_attribute(self.m_attribute, values)

        if status == 1:
            self.m_data_port.add_attribute(self.m_attribute, values, static=False)

        elif status == -1 and self.m_overwrite:
            self.m_data_port.add_attribute(self.m_attribute, values, static=False)

        elif status == -1 and not self.m_overwrite:
            warnings.warn("The attribute '"+str(self.m_attribute)+"' is already present. Set the "
                          "'overwrite' parameter to True in order to overwrite the values with "
                          +str(self.m_file_name)+".")

        elif status == 0:
            warnings.warn("The '"+str(self.m_attribute)+"' attribute is already present and "
                          "contains the same values as are present in "+str(self.m_file_name)+".")

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_data_port.close_port()
