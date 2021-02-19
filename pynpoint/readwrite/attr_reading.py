"""
Modules for reading attributes from a FITS or ASCII file.
"""

import os
import warnings

from typing import Optional

import numpy as np

from astropy.io import fits
from typeguard import typechecked

from pynpoint.core.attributes import get_attributes
from pynpoint.core.processing import ReadingModule


class AttributeReadingModule(ReadingModule):
    """
    Module for reading a list of values from a FITS or ASCII file and appending them as a non-static
    attributes to a dataset.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 data_tag: str,
                 file_name: str,
                 attribute: str,
                 input_dir: Optional[str] = None,
                 overwrite: bool = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        data_tag : str
            Tag of the database entry to which the attribute is written.
        file_name : str
            Name of the input file with the attribute value. Should be equal in size to the number
            of images in ``data_tag``. In case the ``file_name`` is ending with ``.fits``, then a
            FITS file is read. Otherwise, a single column of values is expected in an ASCII file.
        file_name : str
            Name of the input file with a list of values.
        attribute : str
            Name of the attribute as to be written in the database.
        input_dir : str, None
            Input directory where the input file is located. If not specified the Pypeline default
            directory is used.
        overwrite : bool
            Overwrite if the attribute is already exists.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in, input_dir=input_dir)

        self.m_data_port = self.add_output_port(data_tag)

        self.m_file_name = file_name
        self.m_attribute = attribute
        self.m_overwrite = overwrite

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Reads a list of values from a FITS or ASCII file and writes them
        as non-static attribute to a dataset.

        Returns
        -------
        NoneType
            None
        """

        print('Reading attribute data...', end='')

        attributes = get_attributes()

        if self.m_attribute not in attributes:
            raise ValueError(f'\'{self.m_attribute}\' is not a valid attribute.')

        if self.m_file_name.endswith('fits'):
            values = fits.getdata(os.path.join(self.m_input_location, self.m_file_name))
        else:
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


class ParangReadingModule(ReadingModule):
    """
    Module for reading a list of parallactic angles from a FITS or ASCII file.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 data_tag: str,
                 file_name: str,
                 input_dir: Optional[str] = None,
                 overwrite: bool = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        data_tag : str
            Tag of the database entry to which the ``PARANG`` attribute is written.
        file_name : str
            Name of the input file with the parallactic angles (deg). Should be equal in size
            to the number of images in ``data_tag``. In case the ``file_name`` is ending with
            ``.fits``, then a FITS file is read. Otherwise, a single column of values is expected
            in an ASCII file.
        input_dir : str, None
            Input directory where the input file is located. If not specified the Pypeline default
            directory is used.
        overwrite : bool
            Overwrite if the ``PARANG`` attribute already exists.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in, input_dir=input_dir)

        self.m_data_port = self.add_output_port(data_tag)

        self.m_file_name = file_name
        self.m_overwrite = overwrite

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Reads the parallactic angles from a FITS or ASCII file and
        writes the values as non-static attribute (``PARANG``) to the database tag.

        Returns
        -------
        NoneType
            None
        """

        print('Reading parallactic angles...', end='')

        if self.m_file_name.endswith('fits'):
            parang = fits.getdata(os.path.join(self.m_input_location, self.m_file_name))
        else:
            parang = np.loadtxt(os.path.join(self.m_input_location, self.m_file_name))

        print(' [DONE]')

        if parang.ndim != 1:
            raise ValueError(f'The input file {self.m_file_name} should contain a 1D data set with '
                             f'the parallactic angles.')

        print(f'Number of angles: {parang.size}')
        print(f'Rotation range: {parang[0]:.2f} -> {parang[-1]:.2f} deg')

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

        self.m_data_port.close_port()


class WavelengthReadingModule(ReadingModule):
    """
    Module for reading a list of wavelengths from a FITS or ASCII file.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 data_tag: str,
                 file_name: str,
                 input_dir: Optional[str] = None,
                 overwrite: bool = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        data_tag : str
            Tag of the database entry to which the ``WAVELENGTH`` attribute is written.
        file_name : str
            Name of the input file with the wavelengths (a.u.). Should be equal in size
            to the number of images in ``data_tag``. In case the ``file_name`` is ending with
            ``.fits``, then a FITS file is read. Otherwise, a single column of values is expected
            in an ASCII file.
        input_dir : str, None
            Input directory where the input file is located. If not specified the Pypeline default
            directory is used.
        overwrite : bool
            Overwrite if the ``WAVELENGTH`` attribute already exists.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in, input_dir=input_dir)

        self.m_data_port = self.add_output_port(data_tag)

        self.m_file_name = file_name
        self.m_overwrite = overwrite

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Reads the parallactic angles from a FITS or ASCII file and writes
        the values as non-static attribute (``WAVELENGTH``) to the database tag.

        Returns
        -------
        NoneType
            None
        """

        print('Reading wavelengths...', end='')

        if self.m_file_name.endswith('fits'):
            wavelength = fits.getdata(os.path.join(self.m_input_location, self.m_file_name))
        else:
            wavelength = np.loadtxt(os.path.join(self.m_input_location, self.m_file_name))

        print(' [DONE]')

        if wavelength.ndim != 1:
            raise ValueError(f'The input file {self.m_file_name} should contain a 1D data set with '
                             f'the wavelengths.')

        print(f'Number of wavelengths: {wavelength.size}')
        print(f'Wavelength range: {wavelength[0]:.2f} - {wavelength[-1]:.2f}')

        status = self.m_data_port.check_non_static_attribute('WAVELENGTH', wavelength)

        if status == 1:
            self.m_data_port.add_attribute('WAVELENGTH', wavelength, static=False)

        elif status == -1 and self.m_overwrite:
            self.m_data_port.add_attribute('WAVELENGTH', wavelength, static=False)

        elif status == -1 and not self.m_overwrite:
            warnings.warn(f'The WAVELENGTH attribute is already present. Set the \'overwrite\' '
                          f'parameter to True in order to overwrite the values with '
                          f'{self.m_file_name}.')

        elif status == 0:
            warnings.warn(f'The WAVELENGTH attribute is already present and contains the same '
                          f'values as are present in {self.m_file_name}.')

        self.m_data_port.close_port()
