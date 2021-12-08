"""
Modules for writing data as text file.
"""

import os

from typing import Optional

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import WritingModule


class TextWritingModule(WritingModule):
    """
    Module for writing a 1D or 2D data set from the central HDF5 database as text file.
    TextWritingModule is a :class:`pynpoint.core.processing.WritingModule` and supports
    the use of the Pypeline default output directory as well as a specified location.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 data_tag: str,
                 file_name: str,
                 output_dir: Optional[str] = None,
                 header: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        data_tag : str
            Tag of the database entry from which data is exported.
        file_name : str
            Name of the output file.
        output_dir : str, None
            Output directory where the text file will be stored. If no path is specified then the
            Pypeline default output location is used.
        header : str, None
            Header that is written at the top of the text file.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in, output_dir=output_dir)

        self.m_data_port = self.add_input_port(data_tag)

        self.m_file_name = file_name
        self.m_header = header

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Saves the specified data from the database to a text file.

        Returns
        -------
        NoneType
            None
        """

        if self.m_header is None:
            self.m_header = ''

        print('Writing text file...', end='')

        out_name = os.path.join(self.m_output_location, self.m_file_name)

        data = self.m_data_port.get_all()

        if data.ndim == 3 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)

        if data.ndim > 2:
            raise ValueError('Only 1D or 2D arrays can be written to a text file.')

        if data.dtype == 'int32' or data.dtype == 'int64':
            np.savetxt(out_name, data, header=self.m_header, comments='# ', fmt='%i')

        elif data.dtype == 'float32' or data.dtype == 'float64':
            np.savetxt(out_name, data, header=self.m_header, comments='# ')

        print(' [DONE]')

        self.m_data_port.close_port()
