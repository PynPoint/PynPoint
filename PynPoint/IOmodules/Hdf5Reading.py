"""
Module for reading HDF5 files that were created with the Hdf5WritingModule.
"""

import os
import sys

import h5py
import numpy as np

from PynPoint.Core.Processing import ReadingModule
from PynPoint.Util.ModuleTools import progress


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

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param input_filename: The file name of the HDF5 input file. All files inside the input
                               location will be imported if no filename is provided.
        :type input_filename: str
        :param input_dir: The directory of the input HDF5 file. If no location is given, the
                          Pypeline default input location is used.
        :type input_dir: str
        :param tag_dictionary: Directory of all dataset keys / tags which will be imported. The
                               dictionary is used like this:

                               { *tag_of_the_dataset_in_the_hdf5_file* :
                                 *name_of_the_imported_dataset* }

                               All datasets of the external HDF5 file which match one of the
                               *tag_of_the_dataset_in_the_hdf5_file* tags in the tag_dictionary
                               will be imported. Their names inside the internal PynPoint database
                               will be changed to *name_of_the_imported_dataset*.
        :type tag_dictionary: dict

        :return: None
        """

        super(Hdf5ReadingModule, self).__init__(name_in, input_dir)

        if tag_dictionary is None:
            tag_dictionary = {}

        for out_tag in tag_dictionary.itervalues():
            self.add_output_port(out_tag)

        self.m_filename = input_filename
        self._m_tag_dictionary = tag_dictionary

    def _read_single_hdf5(self,
                          file_in):
        """
        Internal function which reads a single HDF5 file.

        :param file_in: name of the HDF5 file.
        :type file_in: str

        :return: None
        """

        hdf5_file = h5py.File(file_in, mode='r')

        for _, entry in enumerate(hdf5_file.keys()):
            # do not read header information groups
            if entry.startswith("header_"):
                continue

            entry = str(entry)  # unicode keys cause errors

            if entry in self._m_tag_dictionary:
                tmp_tag = self._m_tag_dictionary[entry]
            else:
                continue

            # add data
            tmp_port = self._m_output_ports[tmp_tag]
            tmp_port.set_all(np.asarray(hdf5_file[entry][...]))

            # add static attributes
            for attribute_name, attribute_value in hdf5_file[entry].attrs.iteritems():
                tmp_port.add_attribute(name=attribute_name,
                                       value=attribute_value)

            # add non static attributes if existing
            if "header_" + entry in hdf5_file:
                for non_static_attr in hdf5_file[("header_" + entry)]:
                    tmp_port.add_attribute(name=non_static_attr,
                                           value=hdf5_file[("header_" + entry+
                                                            "/"+non_static_attr)][...],
                                           static=False)

    def run(self):
        """
        Run method of the module. Looks for all HDF5 files in the input directory and reads them
        using the internal function _read_single_hdf5().

        :return: None
        """

        # create list of files to be read
        files = []

        tmp_dir = os.path.join(self.m_input_location, '')

        # check if a single input file is given
        if self.m_filename is not None:
            # create file path + filename
            assert(os.path.isfile((tmp_dir + str(self.m_filename)))), "Error: Input file does not "\
                                                                      "exist. Input requested: %s"\
                                                                      % str(self.m_filename)

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
