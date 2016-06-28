import os
import h5py

from PynPoint.Processing import ReadingModule


class Hdf5ReadingModule(ReadingModule):

    def __init__(self,
                 name_in,
                 input_filename = None,
                 input_dir = None,
                 tag_dictionary = {}):

        super(Hdf5ReadingModule, self).__init__(name_in, input_dir)

        self.m_filename = input_filename
        self._m_tag_dictionary = tag_dictionary

    def _read_single_hdf5(self,
                          file_in):
        hdf5_file = h5py.File(file_in, mode='a')

        for entry in hdf5_file.keys():
            # do not read header information groups
            if entry.startswith("header_"):
                continue

            if entry in self._m_tag_dictionary:
                tmp_tag = self._m_tag_dictionary[entry]
            else:
                tmp_tag = entry

            # add data
            self.add_output_port(tmp_tag)
            self._m_out_ports[tmp_tag].set_all(hdf5_file[entry][...])

            # add static attributes
            for attribute_name, attribute_value in hdf5_file[entry].attrs.iteritems():
                self._m_out_ports[tmp_tag].add_attribute(name=attribute_name,
                                                         value=attribute_value)

            # add non static attributes if existing
            if ("header_" + entry) in hdf5_file:
                for non_static_attr in hdf5_file[("header_" + entry)]:
                    self._m_out_ports[tmp_tag].\
                        add_attribute(name=non_static_attr,
                                      value=hdf5_file[("header_" + entry+"/"+non_static_attr)][...],
                                      static=False)

    def run(self):

        # create list of files to be read
        files = []

        if self.m_input_location.endswith("/"):
            tmp_dir = str(self.m_input_location)
        else:
            tmp_dir = str(self.m_input_location) + "/"

        # check is a single input file is given
        if self.m_filename is not None:
            # create file path + filename

            assert(os.path.isfile((tmp_dir + str(self.m_filename)))), "Error: Input file does not "\
                                                                      "exist. Input requested: "\
                                                                      % str(self.m_filename)

            files.append((tmp_dir + str(self.m_filename)))

        else:
            # look for all .hdf5 files in the directory
            for tmp_file in os.listdir(self.m_input_location):
                if tmp_file.endswith('.hdf5') or tmp_file.endswith('.h5'):
                    files.append(tmp_dir + str(tmp_file))

        for tmp_file in files:
            self._read_single_hdf5(tmp_file)
