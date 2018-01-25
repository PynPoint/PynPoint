"""
Module for Writing a list of tags from the database to a separated .hdf5 file
"""

import h5py

from PynPoint.core.Processing import WritingModule


class Hdf5WritingModule(WritingModule):
    """
    Module which exports a part of the PynPoint internal database to a separated .hdf5 file. The
    datasets of the database can be chosen using a tag_dictionary. The module will export static and
    non-static attributes, too.
    """

    def __init__(self,
                 file_name,
                 name_in="hdf5_writing",
                 output_dir=None,
                 tag_dictionary=None,
                 keep_attributes=True):
        """
        Constructor of a Hdf5WritingModule instance.

        :param name_in: Name of the Pypeline module
        :type name_in: str
        :param file_name: Name of the file which will be created by the module.
        :type file_name: str
        :param output_dir: Location where the result .hdf5 will be stored. If no location is given,
                           the Pypeline default output location is used.
        :type output_dir: str
        :param tag_dictionary: Directory containing all tags / keys of the datasets which will be
                               exported from the PynPoint internal database. The datasets will be
                               exported like this:

                                {*tag_of_the_dataset_in_the_PynPoint_database* :
                                *name_of_the_exported_dataset*}
        :type tag_dictionary: dict
        :param keep_attributes: If True all static and non-static attributes will be exported.
        :type keep_attributes: bool
        """

        super(Hdf5WritingModule, self).__init__(name_in, output_dir)

        if tag_dictionary is None:
            tag_dictionary = {}

        self.m_file_name = file_name
        self.m_tag_dictionary = tag_dictionary
        self.m_keep_attributes = keep_attributes

        # Ports will be created on the fly

    def run(self):
        """
        Run method of the module. It exports all datasets defined in the tag_dictionary to an
        external .hdf5 file.

        :return: None
        """

        # create new .hdf5 file
        out_name = os.path.join(self.m_output_location, '') + self.m_file_name
        out_file = h5py.File(out_name, mode='a')

        for in_tag, out_tag in self.m_tag_dictionary.iteritems():

            tmp_port = self.add_input_port(in_tag)

            tmp_data = tmp_port.get_all()

            if tmp_data is None:
                continue

            data_set = out_file.create_dataset(out_tag,
                                               data=tmp_data)

            if self.m_keep_attributes:
                # stable attributes
                tmp_attr = tmp_port.get_all_static_attributes()

                # it is not possible to copy attributes all together
                for key, value in tmp_attr.iteritems():
                    data_set.attrs[key] = value

                # non stable attributes
                non_static_attr_keys = tmp_port.get_all_non_static_attributes()

                if non_static_attr_keys is not None:
                    for key in non_static_attr_keys:
                        tmp_data_attr = tmp_port.get_attribute(key)

                        out_file.create_dataset(("header_" + out_tag + "/" + key),
                                                data=tmp_data_attr)

            tmp_port.close_port()

        out_file.close()
