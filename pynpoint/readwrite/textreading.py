"""
Modules for reading data from a text file.
"""

import os
import warnings

import numpy as np

from typeguard import typechecked

from pynpoint.core.attributes import get_attributes
from pynpoint.core.processing import ReadingModule


class ParangReadingModule(ReadingModule):
    """
    Module for reading a list of parallactic angles from a text file.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 data_tag: str,
                 file_name: str,
                 input_dir: str = None,
                 overwrite: bool = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        data_tag : str
            Tag of the database entry to which the PARANG attribute is written.
        file_name : str
            Name of the input file with a list of parallactic angles (deg). Should be equal in size
            to the number of images in *data_tag*.
        input_dir : str, None
            Input directory where the text file is located. If not specified the Pypeline default
            directory is used.
        overwrite : bool
            Overwrite if the PARANG attribute already exists.

        Returns
        -------
        NoneType
            None
        """
        super(ParangReadingModule, self).__init__(name_in, input_dir)

        self.m_data_port = self.add_output_port(data_tag)

        self.m_file_name = file_name
        self.m_overwrite = overwrite

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Reads the parallactic angles from a text file and writes the
        values as non-static attribute (PARANG) to the database tag.

        Returns
        -------
        NoneType
            None
        """

        print('Reading parallactic angles...', end='')

        parang = np.loadtxt(os.path.join(self.m_input_location, self.m_file_name))

        if parang.ndim != 1:
            raise ValueError(f'The input file {self.m_file_name} should contain a 1D data set with '
                             f'the parallactic angles.')

        status = self.m_data_port.check_non_static_attribute('PARANG', parang)

        if status == 1:
            self.m_data_port.add_attribute('PARANG', parang, static=False)

        elif status == -1 and self.m_overwrite:
            self.m_data_port.add_attribute('PARANG', parang, static=False)

        elif status == -1 and not self.m_overwrite:
            warnings.warn(f'The PARANG attribute is already present. Set the \'overwrite\' '
                          f'parameter to True in order to overwrite the values with '
                          f'{self.m_file_name}.')

        elif status == 0:
            warnings.warn(f'The PARANG attribute is already present and contains the same values '
                          f'as are present in {self.m_file_name}.')

        print(' [DONE]')

        self.m_data_port.close_port()


class AttributeReadingModule(ReadingModule):
    """
    Module for reading a list of values from a text file and appending them as a non-static
    attributes to a dataset.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 data_tag: str,
                 file_name: str,
                 attribute: str,
                 input_dir: str = None,
                 overwrite: bool = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        data_tag : str
            Tag of the database entry to which the attribute is written.
        file_name : str
            Name of the input file with a list of values.
        attribute : str
            Name of the attribute as to be written in the database.
        input_dir : str, None
            Input directory where the text file is located. If not specified the Pypeline default
            directory is used.
        overwrite : bool
            Overwrite if the attribute is already exists.

        Returns
        -------
        NoneType
            None
        """

        super(AttributeReadingModule, self).__init__(name_in, input_dir)

        self.m_data_port = self.add_output_port(data_tag)

        self.m_file_name = file_name
        self.m_attribute = attribute
        self.m_overwrite = overwrite

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Reads a list of values from a text file and writes them as
        non-static attribute to a dataset.

        Returns
        -------
        NoneType
            None
        """

        print('Reading attribute data...', end='')

        attributes = get_attributes()

        if self.m_attribute not in attributes:
            raise ValueError(f'\'{self.m_attribute}\' is not a valid attribute.')

        values = np.loadtxt(os.path.join(self.m_input_location, self.m_file_name),
                            dtype=attributes[self.m_attribute]['type'])

        if values.ndim != 1:
            raise ValueError(f'The input file {self.m_file_name} should contain a 1D list with '
                             f'attributes.')

        status = self.m_data_port.check_non_static_attribute(self.m_attribute, values)

        if status == 1:
            self.m_data_port.add_attribute(self.m_attribute, values, static=False)

        elif status == -1 and self.m_overwrite:
            self.m_data_port.add_attribute(self.m_attribute, values, static=False)

        elif status == -1 and not self.m_overwrite:
            warnings.warn(f'The attribute \'{self.m_attribute}\' is already present. Set the '
                          f'\'overwrite\' parameter to True in order to overwrite the values with '
                          f'{self.m_file_name}.')

        elif status == 0:
            warnings.warn(f'The \'{self.m_attribute}\' attribute is already present and '
                          f'contains the same values as are present in {self.m_file_name}.')

        print(' [DONE]')

        self.m_data_port.close_port()
