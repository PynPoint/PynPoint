"""
Module for reading Hdf5 files that were created using Hdf5WritingModule.
"""

import os
import sys

import h5py
import numpy as np

from PynPoint.core.Processing import ReadingModule
from PynPoint.util.Progress import progress


class Hdf5ReadingModule(ReadingModule):
    """
    ***This class was made for reading HDF5 files created by Hdf5WritingModule. Other data can
    lead to inconsistencies in the PynPoint database.***

    Reads a .hdf5 files from the given input_dir or the default directory of the Pypeline. A tag
    dictionary has to be set in order to choose the datasets which will be imported by the module.
    The module reads static and non-static attributes.
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
        :param input_filename: The input file name. It needs to be a .hdf5 file. If no .hdf5 file is
                               given all files inside the input location will be imported.
        :type input_filename: str
        :param input_dir: The directory of the input .hdf5 file. If no location is given, the
                          Pypeline default input location is used.
        :type input_dir: str
        :param tag_dictionary: Directory of all dataset keys / tags which will be imported. The
                               dictionary is used like this:

                               { *tag_of_the_dataset_in_the_hdf5_file* :
                                 *name_of_the_imported_dataset* }

                               All datasets of the external .hdf5 file which match one of the
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
        Internal function which reads a single .hdf5 file.

        :param file_in: name of the .hdf5 file.
        :type file_in: str

        :return: None
        """
        hdf5_file = h5py.File(file_in, mode='a')

        for i, entry in enumerate(hdf5_file.keys()):
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
        Run method of the module. Looks for all .hdf5 files in the input directory and reads them
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
            # look for all .hdf5 files in the directory
            for tmp_file in os.listdir(self.m_input_location):
                if tmp_file.endswith('.hdf5') or tmp_file.endswith('.h5'):
                    files.append(tmp_dir + str(tmp_file))

        for i, tmp_file in enumerate(files):
            progress(i, len(files), "Running Hdf5ReadingModule...")
            self._read_single_hdf5(tmp_file)

        sys.stdout.write("Running Hdf5ReadingModule... [DONE]\n")
        sys.stdout.flush()
