"""
Module for reading HDF5 files that were created with the
:class:`~pynpoint.readwrite.hdf5writing.Hdf5WritingModule`.
"""

import os
import time
import warnings

import h5py
import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ReadingModule
from pynpoint.util.module import progress


class Hdf5ReadingModule(ReadingModule):
    """
    Reads an HDF5 file from the given *input_dir* or the default directory of the Pypeline. A tag
    dictionary has to be set in order to choose the datasets which will be imported into the
    database. Also the static and non-static attributes are read from the HDF5 file and stored
    in the database with the corresponding data set. This module should only be used for reading
    HDF5 files that are created with the Hdf5WritingModule. Reading different type of HDF5 files
    may lead to inconsistencies in the central database.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 input_filename: str = None,
                 input_dir: str = None,
                 tag_dictionary: dict = None):
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        input_filename : str, None
            The file name of the HDF5 input file. All files inside the input location will be
            imported if no filename is provided.
        input_dir : str, None
            The directory of the input HDF5 file. If no location is given, the default input
            location of the Pypeline is used.
        tag_dictionary : dict, None
            Dictionary of all data sets that will be imported. The dictionary format is
            {*tag_name_in_input_file*:*tag_name_in_database*, }. All data sets in the input HDF5
            file that match one of the *tag_name_in_input_file* will be imported. The tag name
            inside the internal Pypeline database will be changed to *tag_name_in_database*.

        Returns
        -------
        NoneType
            None
        """

        super(Hdf5ReadingModule, self).__init__(name_in, input_dir)

        if tag_dictionary is None:
            tag_dictionary = {}

        for out_tag in tag_dictionary.values():
            self.add_output_port(out_tag)

        self.m_filename = input_filename
        self._m_tag_dictionary = tag_dictionary

    @typechecked
    def read_single_hdf5(self,
                         file_in: str) -> None:
        """
        Function which reads a single HDF5 file.

        Parameters
        ----------
        file_in : str
            Path and name of the HDF5 file.

        Returns
        -------
        NoneType
            None
        """

        hdf5_file = h5py.File(file_in, mode='r')

        for tag_in in self._m_tag_dictionary:
            tag_in = str(tag_in)  # unicode keys cause errors
            tag_out = self._m_tag_dictionary[tag_in]

            if tag_in not in hdf5_file:
                warnings.warn(f'The dataset with tag name \'{tag_in}\' is not found in the HDF5 '
                              f'file.')

                continue

            # add data
            port_out = self._m_output_ports[tag_out]
            port_out.set_all(np.asarray(hdf5_file[tag_in][...]))

            # add static attributes
            for attr_name, attr_value in hdf5_file[tag_in].attrs.items():
                port_out.add_attribute(name=attr_name, value=attr_value)

            # add non-static attributes
            if 'header_' + tag_in in hdf5_file:
                for attr_name in hdf5_file['header_' + tag_in]:
                    attr_val = hdf5_file['header_' + tag_in + '/' + attr_name][...]
                    port_out.add_attribute(name=attr_name, value=attr_val, static=False)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Looks for all HDF5 files in the input directory and reads the
        datasets that are provided in the tag dictionary.

        Returns
        -------
        NoneType
            None
        """

        # create list of files to be read
        files = []

        tmp_dir = os.path.join(self.m_input_location, '')

        # check if a single input file is given
        if self.m_filename is not None:
            # create file path + filename
            assert(os.path.isfile((tmp_dir + str(self.m_filename)))), \
                   f'Error: Input file does not exist. Input requested: {self.m_filename}'

            files.append((tmp_dir + str(self.m_filename)))

        else:
            # look for all HDF5 files in the directory
            for tmp_file in os.listdir(self.m_input_location):
                if tmp_file.endswith('.hdf5') or tmp_file.endswith('.h5'):
                    files.append(tmp_dir + str(tmp_file))

        start_time = time.time()
        for i, tmp_file in enumerate(files):
            progress(i, len(files), 'Reading HDF5 file...', start_time)
            self.read_single_hdf5(tmp_file)
