"""
Module for Writing a list of tags from the database to a separated .hdf5 file
"""

import h5py

from PynPoint.core.Processing import WritingModule


class Hdf5WritingModule(WritingModule):

    def __init__(self,
                 name_in,
                 file_name,
                 output_dir=None,
                 tag_dictionary=None,
                 keep_attributes = True):

        super(Hdf5WritingModule, self).__init__(name_in, output_dir)
        self.m_file_name = file_name
        self.m_tag_dictionary = tag_dictionary
        self.m_keep_attributes = keep_attributes

        # Add Ports
        for key in tag_dictionary:
            self.add_input_port(key)

    def run(self):

        # create new .hdf5 file
        out_file = h5py.File((self.m_output_location + '/' + self.m_file_name), mode='a')

        for in_tag, out_tag in self.m_tag_dictionary.iteritems():

            tmp_data = self._m_input_ports[in_tag].get_all()

            if tmp_data is None:
                continue

            data_set = out_file.create_dataset(out_tag,
                                               data=tmp_data)

            if self.m_keep_attributes:
                # stable attributes
                tmp_attr = self._m_input_ports[in_tag].get_all_static_attributes()

                # it is not possible to copy attributes all together
                for key, value in tmp_attr.iteritems():
                    data_set.attrs[key] = value

                # non stable attributes
            non_static_attr_keys = self._m_input_ports[in_tag].get_all_non_static_attributes()

            if non_static_attr_keys is not None:
                for key in non_static_attr_keys:
                    tmp_data_attr = self._m_input_ports[in_tag].get_attribute(key)

                    out_file.create_dataset(("header_" + out_tag + "/" + key),
                                            data=tmp_data_attr)

        out_file.close()

