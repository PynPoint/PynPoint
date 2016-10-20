import os
import pytest
import numpy as np
import warnings

from PynPoint.core.DataIO import OutputPort, DataStorage, InputPort

import warnings
warnings.simplefilter("always")


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
        # ---- Test database not set -----

        data = [1, 2, 3, 4, 0]

        with pytest.warns(UserWarning) as record:
            out_port = OutputPort("some_data")
            out_port.set_all(data)

        # check that only one warning was raised
        assert len(record) == 1
        # check that the message matches
        assert record[0].message.args[0] == "Port can not store data unless a database is connected"

        # ---- Test data dim of actual data for new data entry is < 1 or > 3

        _, out_port = self.create_input_and_output_port("new_data")

        data = [[[[2, 2], ], ], ]

        with pytest.raises(ValueError) as ex_info:

            out_port.set_all(data, data_dim=2)

        assert ex_info.value.message == 'Output port can only save numpy arrays from 1D to 3D. If '\
                                        'you want to save a int, float, string ... use Port' \
                                        ' attributes instead.'

        # ---- Test data dim of data_dim for new data entry is < 1 or > 3

        _, out_port = self.create_input_and_output_port("new_data")

        data = [1, 2, 4]

        with pytest.raises(ValueError) as ex_info:

            out_port.set_all(data, data_dim=0)

        assert ex_info.value.message == 'data_dim needs to be in [1,3].'

        # ---- Test data_dim for new data entry is smaller than actual data

        _, out_port = self.create_input_and_output_port("new_data")

        data = [[1], [2]]

        with pytest.raises(ValueError) as ex_info:

            out_port.set_all(data, data_dim=1)

        assert ex_info.value.message == 'data_dim needs to have at least the same dim as the input.'

        # ---- Test data_dim == 3 and actual size == 1

        _, out_port = self.create_input_and_output_port("new_data")

        data = [1, 2]

        with pytest.raises(ValueError) as ex_info:

            out_port.set_all(data, data_dim=3)

        assert ex_info.value.message == 'Cannot initialize 1D data in 3D data container.'

    def test_set_all_keep_attributes(self):

        def init_out_port():
            control, out_port = self.create_input_and_output_port("new_data")

            data = [2, 3, 4]
            out_port.set_all(data)
            out_port.add_attribute(name="test1",
                                   value=1)
            out_port.add_attribute(name="test2",
                                   value=12)
            return out_port, control

        out_port, control = init_out_port()
        out_port.set_all([[]],
                         data_dim=2,
                         keep_attributes=True)

        assert control.get_attribute("test1") == 1
        assert control.get_attribute("test2") == 12
        out_port.del_all_data()

    def test_append_new_data(self):
        # using append even if no data exists
        control, out_port = self.create_input_and_output_port("new_data")
        # ----- 1D input -----
        data = [3, ]
        out_port.append(data)

        assert control.get_all() == data
        out_port.del_all_data()

        # ----- 2D input -----
        data = [[3, 3], [3, 2]]
        out_port.append(data)

        assert np.array_equal(control.get_all(), data)
        out_port.del_all_data()

        # ----- 3D input -----
        data = [[[3, 3], [3, 2]], [[3, 1], [3, 1]]]
        out_port.append(data)

        assert np.array_equal(control.get_all(), data)
        out_port.del_all_data()

    def test_append_existing_data(self):
        control, out_port = self.create_input_and_output_port("new_data")

        # ----- 1D -----
        out_port.append([2, 3, 5], data_dim=1)
        out_port.append([3, 3, 5])
        assert np.array_equal(control.get_all(),
                              [2, 3, 5, 3, 3, 5])
        out_port.del_all_data()

        # ----- 2D -----
        # 1D input append to 1D data
        out_port.append([1, 1], data_dim=2)
        out_port.append([3, 3])
        assert np.array_equal(control.get_all(),
                              [[1, 1], [3, 3]])
        out_port.del_all_data()

        # 1D input append to 2D data
        out_port.append([[2, 3], [1, 1]], data_dim=2)
        out_port.append([3, 3])
        assert np.array_equal(control.get_all(),
                              [[2, 3], [1, 1], [3, 3]])
        out_port.del_all_data()

        # 2D input append to 2D data
        out_port.append([[2, 3], [1, 1]], data_dim=2)
        out_port.append([[3, 3], [8, 8]])
        assert np.array_equal(control.get_all(),
                              [[2, 3], [1, 1], [3, 3], [8, 8]])
        out_port.del_all_data()

        # 2D input append to 3D data
        out_port.append([[[2, 3], [1, 1]],
                         [[2, 4], [1, 1]]], data_dim=3)
        out_port.append([[3, 3], [8, 8]])
        assert np.array_equal(control.get_all(),
                              [[[2, 3], [1, 1]],
                               [[2, 4], [1, 1]],
                               [[3, 3], [8, 8]]])
        out_port.del_all_data()

        # 3D input append to 3D data
        out_port.append([[[2, 3], [1, 1]],
                         [[2, 4], [1, 1]]], data_dim=3)
        out_port.append([[[22, 7], [10, 221]],
                         [[223, 46], [1, 15]]])

        assert np.array_equal(control.get_all(),
                              [[[2, 3], [1, 1]],
                               [[2, 4], [1, 1]],
                               [[22, 7], [10, 221]],
                               [[223, 46], [1, 15]]])
        out_port.del_all_data()

    def test_append_existing_data_force_overwriting(self):
        control, out_port = self.create_input_and_output_port("new_data")

        # Error case (no force)
        out_port.append([2, 3, 5], data_dim=1)

        out_port.append([[[22, 7], [10, 221]],
                         [[223, 46], [1, 15]]],
                        force=True)

        assert np.array_equal(control.get_all(), [[[22, 7], [10, 221]],
                                                  [[223, 46], [1, 15]]])
        out_port.del_all_data()

    def test_append_existing_data_error(self):
        # ---- port not active ----
        control, out_port = self.create_input_and_output_port("new_data")
        out_port.deactivate()
        data = [1, ]
        out_port.append(data)
        out_port.del_all_data()
        out_port.activate()

        # 1 Element input

        # 1D input
        # 2D input
        # 3D input

        # Error case (no force)
        out_port.set_all([2, 3, 5], data_dim=1)

        with pytest.raises(ValueError) as ex_info:
            out_port.append([[[22, 7], [10, 221]],
                             [[223, 46], [1, 15]]])

        assert ex_info.value.message == 'The port tag new_data is already used with a different ' \
                                        'data type. If you want to replace it use force = True.'
        out_port.del_all_data()

    def test_set_data_using_slicing(self):
        control, out_port = self.create_input_and_output_port("new_data")
        out_port.set_all([2, 5, 6, 7, ])
        out_port[3] = 44
        assert np.array_equal(control.get_all(),
                              [2, 5, 6, 44, ])
        out_port.deactivate()
        out_port[2] = 0
        assert np.array_equal(control.get_all(),
                              [2, 5, 6, 44, ])
        out_port.activate()
        out_port.del_all_data()

    def test_del_all_data(self):
        control, out_port = self.create_input_and_output_port("new_data")
        out_port.set_all([0, 1])
        out_port.del_all_data()

        assert control.get_all() is None

    def test_add_static_attribute(self):
        control, out_port = self.create_input_and_output_port("new_data")
        out_port.set_all([1])

        out_port.add_attribute("attr1",
                               value=5)
        out_port.add_attribute("attr2",
                               value="no")

        assert control.get_attribute("attr1") == 5

        # update
        out_port.add_attribute("attr1",
                               value=6)
        assert control.get_attribute("attr1") == 6
        assert control.get_attribute("attr2") == "no"

        out_port.deactivate()
        out_port.add_attribute("attr3",
                               value=33)

        assert control.get_attribute("attr3") is None
        out_port.activate()
        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_add_static_attribute_error(self):
        control, out_port = self.create_input_and_output_port("new_data")

        # add attribute while no data is set
        with pytest.warns(UserWarning) as warning:
            out_port.add_attribute("attr1",
                                   value=6)

        # check that only one warning was raised
        assert len(warning) == 1
        # check that the message matches
        assert warning[0].message.args[0] == "Can not save attribute while no data exists."

        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_add_non_static_attribute(self):
        # two different data types
        control, out_port = self.create_input_and_output_port("new_data")
        out_port.set_all([1])

        out_port.add_attribute("attr1",
                               value=[6, 3],
                               static=False)

        assert np.array_equal(control.get_attribute("attr1"),
                              [6, 3])

        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_append_attribute_data(self):
        control, out_port = self.create_input_and_output_port("new_data")
        out_port.del_all_data()
        out_port.set_all([1])

        out_port.add_attribute("attr1",
                               value=[2, 3],
                               static=False)

        assert np.array_equal(control.get_attribute("attr1"),
                              [2, 3])

        out_port.append_attribute_data("attr1",
                                       value=2)

        assert np.array_equal(control.get_attribute("attr1"),
                              [2, 3, 2])

        out_port.deactivate()
        out_port.append_attribute_data("attr1",
                                       value=2)

        assert np.array_equal(control.get_attribute("attr1"),
                              [2, 3, 2])

        out_port.activate()
        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_add_value_to_static_attribute(self):
        control, out_port = self.create_input_and_output_port("new_data")
        out_port.set_all([1])

        out_port.add_attribute("attr1",
                               value=4)

        assert control.get_attribute("attr1") == 4
        out_port.add_value_to_static_attribute("attr1",
                                               value=2)
        assert control.get_attribute("attr1") == 6

        out_port.deactivate()
        out_port.add_value_to_static_attribute("attr1",
                                               value=2)
        assert control.get_attribute("attr1") == 6

        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_add_value_to_static_attribute_error(self):
        # add non int or float data
        control, out_port = self.create_input_and_output_port("new_data")
        out_port.set_all([1])

        out_port.add_attribute("attr1",
                               value="bla")

        assert control.get_attribute("attr1") == "bla"
        with pytest.raises(ValueError) as error:
            out_port.add_value_to_static_attribute("attr1",
                                                   value="bla")

        assert error.value.message == "Can only add integer and float values to an existing " \
                                      "attribute"

        # add data to not existing attribute
        with pytest.raises(AttributeError) as error2:
            out_port.add_value_to_static_attribute("attr42",
                                                   value=3)
        assert error2.value.message == "Can not add value to not existing attribute"

        out_port.del_all_attributes()
        out_port.del_all_data()

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
