"""
Module for writing data as text file.
"""

import os
import sys

import numpy as np

from PynPoint.core.Processing import WritingModule


class TextWritingModule(WritingModule):
    """
    Module for writing a 1D or 2D data set from the central .hdf5 database as text file.
    TextWritingModule is a WritingModule and supports to use the Pypeline default output
    directory as well as an own location. See :class:`PynPoint.core.Processing.WritingModule`
    for more information.
    """

    def __init__(self,
                 file_name,
                 name_in="text_writing",
                 output_dir=None,
                 data_tag="im_arr",
                 header=None):
        """
        Constructor of TextWritingModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param file_name: Name of the output file.
        :type file_name: str
        :param output_dir: Output directory where the text file will be stored. If no folder is
                           specified the Pypeline default is chosen.
        :type output_dir: str
        :param data_tag: Tag of the data base entry the module has to export as text file.
        :type data_tag: str
        :param header: Header that is written at the top of the text file.
        :type header: str

        :return: None
        """
        super(TextWritingModule, self).__init__(name_in=name_in, output_dir=output_dir)

        if not isinstance(file_name, str):
            raise ValueError("Output file_name needs to be a string.")

        self.m_data_port = self.add_input_port(data_tag)

        self.m_file_name = file_name
        self.m_header = header

    def run(self):
        """
        Run method of the module. Saves the specified data to a text file.

        :return: None
        """

        if self.m_header is None:
            self.m_header = ""

        sys.stdout.write("Running TextWritingModule...")
        sys.stdout.flush()

        out_name = os.path.join(self.m_output_location, self.m_file_name)

        data = self.m_data_port.get_all()

        if data.ndim > 2:
            raise ValueError("Only 1D or 2D arrays can be written to a text file.")

        np.savetxt(out_name, data, header=self.m_header, comments='# ')

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_data_port.close_port()
