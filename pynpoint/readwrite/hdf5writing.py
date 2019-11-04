"""
Module for writing a list of tags from the database to a separate HDF5 file.
"""

import os

import h5py
from typeguard import typechecked

from pynpoint.core.processing import WritingModule


class Hdf5WritingModule(WritingModule):
    """
    Module which exports a part of the PynPoint internal database to a separate HDF5 file. The
    datasets of the database can be chosen using the *tag_dictionary*. The module will also export
    the static and non-static attributes.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 file_name: str,
                 output_dir: str = None,
                 tag_dictionary: dict = None,
                 keep_attributes: bool = True,
                 overwrite: bool = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        file_name : str
            Name of the file which will be created by the module.
        output_dir : str, None
            Location where the HDF5 file will be stored. The Pypeline default output location is
            used when no location is given.
        tag_dictionary : dict, None
            Directory containing all tags / keys of the datasets which will be exported from the
            PynPoint internal database. The datasets will be exported as
            {*input_tag*:*output_tag*, }.
        keep_attributes : bool
            If True all static and non-static attributes will be exported.
        overwrite : bool
            Overwrite an existing HDF5 file.

        Returns
        -------
        NoneType
            None
        """

        super(Hdf5WritingModule, self).__init__(name_in, output_dir)

        if tag_dictionary is None:
            tag_dictionary = {}

        self.m_file_name = file_name
        self.m_tag_dictionary = tag_dictionary
        self.m_keep_attributes = keep_attributes
        self.m_overwrite = overwrite

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Exports all datasets defined in the *tag_dictionary* to an
        external HDF5 file.

        Returns
        -------
        NoneType
            None
        """

        print('Writing HDF5 file...', end='')

        if self.m_overwrite:
            out_file = h5py.File(os.path.join(self.m_output_location, self.m_file_name), mode='w')
        else:
            out_file = h5py.File(os.path.join(self.m_output_location, self.m_file_name), mode='a')

        for in_tag, out_tag in self.m_tag_dictionary.items():
            tmp_port = self.add_input_port(in_tag)
            tmp_data = tmp_port.get_all()

            if tmp_data is None:
                continue

            data_set = out_file.create_dataset(out_tag, data=tmp_data)

            if self.m_keep_attributes:
                # static attributes
                tmp_attr = tmp_port.get_all_static_attributes()

                # it is not possible to copy attributes all together
                for key, value in tmp_attr.items():
                    data_set.attrs[key] = value

                # non-static attributes
                non_static_attr_keys = tmp_port.get_all_non_static_attributes()

                if non_static_attr_keys is not None:
                    for key in non_static_attr_keys:
                        tmp_data_attr = tmp_port.get_attribute(key)
                        attr_tag = 'header_' + out_tag + '/' + key
                        out_file.create_dataset(attr_tag, data=tmp_data_attr)

            tmp_port.close_port()

        out_file.close()

        print(' [DONE]')
