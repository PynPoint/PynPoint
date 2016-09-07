import os
import pytest
import numpy as np

from PynPoint.core.DataIO import OutputPort, DataStorage, InputPort


class TestOutputPort(object):

    def setup(self):
        self.test_data_dir = (os.path.dirname(__file__)) + '/test_data/'

        dir_in = self.test_data_dir + "init/PynPoint_database.hdf5"
        self.storage = DataStorage(dir_in)

    def create_input_and_output_port(self,
                                     tag_name):

        inport = InputPort(tag_name,
                           self.storage)
        outport = OutputPort(tag_name,
                             self.storage)

        inport.open_port()

        return inport, outport

    def test_create_instance(self):

        active_port = OutputPort("test",
                                 self.storage,
                                 activate_init=True)

        deactive_port = OutputPort("test",
                                   self.storage,
                                   activate_init=False)

        control_port = InputPort("test",
                                 self.storage)

        deactive_port.open_port()

        deactive_port.set_all(np.asarray([0, 1, 2, 3]))
        deactive_port.flush()

        print control_port.get_all()

        assert not np.array_equal(np.asarray([0, 1, 2, 3]),
                                  control_port.get_all())

        active_port.set_all(np.asarray([0, 1, 2, 3]))
        active_port.flush()

        assert np.array_equal(np.asarray([0, 1, 2, 3]),
                              control_port.get_all())

        active_port.del_all_data()

    def test_set_all_new_data(self):

        inport, outport = self.create_input_and_output_port("new_data")

        # ----- 1D input -----
        data = [1, 3]

        outport.set_all(data,
                        data_dim=1)

        assert np.array_equal(inport.get_all(),
                              [1., 3.])
        outport.del_all_data()

        data = [1, 3]

        outport.set_all(data,
                        data_dim=2)

        assert np.array_equal(inport.get_all(),
                              [[1, 3]])

        outport.del_all_data()

        # ----- 2D input -----
        data = [[1, 3], [2, 4]]

        outport.set_all(data,
                        data_dim=2)

        assert np.array_equal(inport.get_all(),
                              [[1, 3], [2, 4]])
        outport.del_all_data()

        data = [[1, 3], [2, 4]]

        outport.set_all(data,
                        data_dim=3)

        assert np.array_equal(inport.get_all(),
                              [[[1, 3], [2, 4]]])

        outport.del_all_data()

        # ----- 3D input -----
        data = [[[1, 3], [2, 4]],[[1, 3], [2, 4]]]

        outport.set_all(data,
                        data_dim=3)

        assert np.array_equal(inport.get_all(),
                              [[[1, 3], [2, 4]], [[1, 3], [2, 4]]])

        outport.del_all_data()


    def test_set_all_error(self):
        pass

    def test_set_all_keep_attributes(self):
        pass

    def test_set_all_new_data_existing_data(self):
        # 1 Element input
        # 1D input
        # 2D input
        # 3D input
        pass

    def test_append_new_data(self):
        # 1 Element input
        # 1D input
        # 2D input
        # 3D input
        pass

    def test_append_existing_data(self):
        # 1 Element input
        # 1D input
        # 2D input
        # 3D input
        pass

    def test_append_existing_data_error(self):
        # 1 Element input
        # 1D input
        # 2D input
        # 3D input
        pass

    def test_activate_deactivate(self):
        pass

    def test_add_static_attribute(self):
        pass

    def test_add_static_attribute_error(self):
        # add array
        pass

    def test_add_non_static_attribute(self):
        # two different data types
        pass

    def test_append_attribute_data(self):
        pass

    def test_append_attribute_data_error(self):
        pass

    def test_append_add_value_to_static_attribute(self):
        pass

    def test_add_value_to_static_attribute_error(self):
        pass

    def test_copy_attributes_from_input_port(self):
        pass

    def test_copy_attributes_from_input_port_error(self):
        pass

    def test_del_attribute(self):
        pass

    def test_del_attribute_not_existing(self):
        pass

    def test_del_all_attributes(self):
        pass

    def test_add_history_information(self):
        pass

    def test_flush(self):
        pass

    def test_check_static_attribute(self):
        pass
