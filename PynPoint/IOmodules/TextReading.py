"""
Modules for reading data from a text file.
"""

import os
import sys
import warnings

import numpy as np

from PynPoint.Core.Processing import ReadingModule


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

        :param file_name: Name of the input file with a list of parallactic angles (deg). Should
                          be equal in size to the number of images in *data_tag*.
        :type file_name: str
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param input_dir: Input directory where the text file is located. If not specified the
                          Pypeline default directory is used.
        :type input_dir: str
        :param data_tag: Tag of the database entry to which the NEW_PARA attribute is written.
        :type data_tag: str
        :param overwrite: Overwrite if the NEW_PARA attribute already exists.
        :type overwrite: bool

        :return: None
        """
        super(ParangReadingModule, self).__init__(name_in=name_in, input_dir=input_dir)

        if not isinstance(file_name, str):
            raise ValueError("Output file_name needs to be a string.")

        self.m_data_port = self.add_output_port(data_tag)

        self.m_file_name = file_name
        self.m_overwrite = overwrite

    def run(self):
        """
        Run method of the module. Reads the parallactic angles from a text file and writes the
        values as non-static attribute (NEW_PARA) to the database tag.

        :return: None
        """

        sys.stdout.write("Running ParangReadingModule...")
        sys.stdout.flush()

        parang = np.loadtxt(os.path.join(self.m_input_location, self.m_file_name))

        if parang.ndim != 1:
            raise ValueError("The input file %s should contain a 1D data set with the parallactic "
                             "angles." % self.m_file_name)

        status = self.m_data_port.check_non_static_attribute("NEW_PARA", None)

        if status == 1:
            self.m_data_port.add_attribute("NEW_PARA", parang, static=False)

        elif status == -1 and self.m_overwrite:
            self.m_data_port.add_attribute("NEW_PARA", parang, static=False)

        elif status == -1 and not self.m_overwrite:
            warnings.warn("The NEW_PARA attribute is already present. Set the overwrite argument "
                          "to True in order to overwrite the values with "+str(self.m_file_name)+
                          ".")

        elif status == 0:
            warnings.warn("The NEW_PARA attribute is already present and contains the same values "
                          "as are present in "+str(self.m_file_name)+".")

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_data_port.close_port()
