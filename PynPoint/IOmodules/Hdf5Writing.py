"""
Module for writing a list of tags from the database to a separate HDF5 file.
"""

from __future__ import absolute_import

import os
import sys

import six
import h5py

from PynPoint.Core.Processing import WritingModule


class Hdf5WritingModule(WritingModule):
    """
    Module which exports a part of the PynPoint internal database to a separate HDF5 file. The
    datasets of the database can be chosen using the *tag_dictionary*. The module will also export
    the static and non-static attributes.
    """

    def __init__(self,
                 file_name,
                 name_in="hdf5_writing",
                 output_dir=None,
                 tag_dictionary=None,
                 keep_attributes=True,
                 overwrite=False):
        """
        Constructor of Hdf5WritingModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param file_name: Name of the file which will be created by the module.
        :type file_name: str
        :param output_dir: Location where the HDF5 file will be stored. The Pypeline default
                           output location is used when no location is given.
        :type output_dir: str
        :param tag_dictionary: Directory containing all tags / keys of the datasets which will be
                               exported from the PynPoint internal database. The datasets will be
                               exported like this:

                                { *PynPoint_database_tag* : *output_tag* }
        :type tag_dictionary: dict
        :param keep_attributes: If True all static and non-static attributes will be exported.
        :type keep_attributes: bool
        :param overwrite: Overwrite an existing HDF5 file.
        :type overwrite: bool

        :return: None
        """

        super(Hdf5WritingModule, self).__init__(name_in, output_dir)

        if tag_dictionary is None:
            tag_dictionary = {}

        self.m_file_name = file_name
        self.m_tag_dictionary = tag_dictionary
        self.m_keep_attributes = keep_attributes
        self.m_overwrite = overwrite

    def run(self):
        """
        Run method of the module. Exports all datasets defined in the *tag_dictionary* to an
        external HDF5 file.

        :return: None
        """

        sys.stdout.write("Running Hdf5WritingModule...")
        sys.stdout.flush()

        if self.m_overwrite:
            out_file = h5py.File(os.path.join(self.m_output_location, self.m_file_name), mode='w')
        else:
            out_file = h5py.File(os.path.join(self.m_output_location, self.m_file_name), mode='a')

        for in_tag, out_tag in six.iteritems(self.m_tag_dictionary):
            tmp_port = self.add_input_port(in_tag)
            tmp_data = tmp_port.get_all()

            if tmp_data is None:
                continue

            data_set = out_file.create_dataset(out_tag, data=tmp_data)

            if self.m_keep_attributes:
                # static attributes
                tmp_attr = tmp_port.get_all_static_attributes()

                # it is not possible to copy attributes all together
                for key, value in six.iteritems(tmp_attr):
                    data_set.attrs[key] = value

                # non-static attributes
                non_static_attr_keys = tmp_port.get_all_non_static_attributes()

                if non_static_attr_keys is not None:
                    for key in non_static_attr_keys:
                        tmp_data_attr = tmp_port.get_attribute(key)
                        attr_tag = "header_" + out_tag + "/" + key
                        out_file.create_dataset(attr_tag, data=tmp_data_attr)

            tmp_port.close_port()

        out_file.close()

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()
