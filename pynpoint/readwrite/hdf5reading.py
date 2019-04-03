"""
Module for reading HDF5 files that were created with the Hdf5WritingModule.
"""

from __future__ import absolute_import

import os
import sys
import warnings

import six
import h5py
import numpy as np

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

    def __init__(self,
                 name_in="hdf5_reading",
                 input_filename=None,
                 input_dir=None,
                 tag_dictionary=None):
        """
        Constructor of Hdf5ReadingModule.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        input_filename : str
            The file name of the HDF5 input file. All files inside the input location will be
            imported if no filename is provided.
        input_dir : str
            The directory of the input HDF5 file. If no location is given, the default input
            location of the Pypeline is used.
        tag_dictionary : dict
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

        for out_tag in six.itervalues(tag_dictionary):
            self.add_output_port(out_tag)

        self.m_filename = input_filename
        self._m_tag_dictionary = tag_dictionary

    def _read_single_hdf5(self,
                          file_in):
        """
        Internal function which reads a single HDF5 file.

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
            tag_in = str(tag_in) # unicode keys cause errors
            tag_out = self._m_tag_dictionary[tag_in]

            if tag_in not in hdf5_file:
                warnings.warn("The dataset with tag name '{0}' is not found in the HDF5 file."
                              .format(tag_in))
                continue

            # add data
            port_out = self._m_output_ports[tag_out]
            port_out.set_all(np.asarray(hdf5_file[tag_in][...]))

            # add static attributes
            for attr_name, attr_value in six.iteritems(hdf5_file[tag_in].attrs):
                port_out.add_attribute(name=attr_name, value=attr_value)

            # add non-static attributes
            if "header_" + tag_in in hdf5_file:
                for attr_name in hdf5_file["header_" + tag_in]:
                    attr_val = hdf5_file["header_" + tag_in + "/" + attr_name][...]
                    port_out.add_attribute(name=attr_name, value=attr_val, static=False)

    def run(self):
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
                   "Error: Input file does not exist. Input requested: %s" % str(self.m_filename)

            files.append((tmp_dir + str(self.m_filename)))

        else:
            # look for all HDF5 files in the directory
            for tmp_file in os.listdir(self.m_input_location):
                if tmp_file.endswith('.hdf5') or tmp_file.endswith('.h5'):
                    files.append(tmp_dir + str(tmp_file))

        for i, tmp_file in enumerate(files):
            progress(i, len(files), "Running Hdf5ReadingModule...")
            self._read_single_hdf5(tmp_file)

        sys.stdout.write("Running Hdf5ReadingModule... [DONE]\n")
        sys.stdout.flush()
