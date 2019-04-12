from __future__ import absolute_import

import os
import warnings

import pytest
import numpy as np

from pynpoint.core.dataio import OutputPort, DataStorage, InputPort
from pynpoint.util.tests import create_random

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    create_random(os.path.dirname(__file__))

def teardown_module():
    os.remove(os.path.dirname(__file__) + "/PynPoint_database.hdf5")

class TestOutputPort(object):

    def setup(self):
        self.storage = DataStorage(os.path.dirname(__file__) + "/PynPoint_database.hdf5")

    def create_input_port(self, tag_name):
        inport = InputPort(tag_name, self.storage)
        inport.open_port()

        return inport

    def create_output_port(self, tag_name):
        outport = OutputPort(tag_name, self.storage)

        return outport

    def test_create_instance(self):
        with pytest.raises(ValueError) as error:
            OutputPort("config", self.storage)

        assert str(error.value) == "The tag name 'config' is reserved for the central " \
                                   "configuration of PynPoint."

        with pytest.raises(ValueError) as error:
            OutputPort("fits_header", self.storage)

        assert str(error.value) == "The tag name 'fits_header' is reserved for storage of the " \
                                   "FITS headers."

        active_port = OutputPort("test", self.storage, activate_init=True)
        deactive_port = OutputPort("test", self.storage, activate_init=False)
        control_port = InputPort("test", self.storage)

        deactive_port.open_port()
        deactive_port.set_all(np.asarray([0, 1, 2, 3]))
        deactive_port.flush()

        with pytest.warns(UserWarning) as warning:
            control_port.get_all()

        assert len(warning) == 1
        assert warning[0].message.args[0] == "No data under the tag which is linked by the " \
                                             "InputPort."

        active_port.set_all(np.asarray([0, 1, 2, 3]))
        active_port.flush()

        assert np.array_equal(np.asarray([0, 1, 2, 3]), control_port.get_all())

        active_port.del_all_data()

    def test_set_all_new_data(self):
        outport = self.create_output_port("new_data")

        # ----- 1D input -----

        data = [1, 3]
        outport.set_all(data, data_dim=1)

        inport = self.create_input_port("new_data")

        assert np.array_equal(inport.get_all(), [1., 3.])
        outport.del_all_data()

        data = [1, 3]
        outport.set_all(data, data_dim=2)
        assert np.array_equal(inport.get_all(), [[1, 3]])
        outport.del_all_data()

        # ----- 2D input -----

        data = [[1, 3], [2, 4]]
        outport.set_all(data, data_dim=2)
        assert np.array_equal(inport.get_all(), [[1, 3], [2, 4]])
        outport.del_all_data()

        data = [[1, 3], [2, 4]]
        outport.set_all(data, data_dim=3)
        assert np.array_equal(inport.get_all(), [[[1, 3], [2, 4]]])
        outport.del_all_data()

        # ----- 3D input -----

        data = [[[1, 3], [2, 4]], [[1, 3], [2, 4]]]
        outport.set_all(data, data_dim=3)
        assert np.array_equal(inport.get_all(), [[[1, 3], [2, 4]], [[1, 3], [2, 4]]])
        outport.del_all_data()

    def test_set_all_error(self):
        # ---- Test database not set -----
        data = [1, 2, 3, 4, 0]

        with pytest.warns(UserWarning) as record:
            out_port = OutputPort("some_data")
            out_port.set_all(data)

        assert len(record) == 1
        assert record[0].message.args[0] == "OutputPort can not store data unless a database is " \
                                            "connected."

        # ---- Test data dim of actual data for new data entry is < 1 or > 3

        out_port = self.create_output_port("new_data")

        data = [[[[2, 2], ], ], ]

        with pytest.raises(ValueError) as error:
            out_port.set_all(data, data_dim=2)

        assert str(error.value) == 'Output port can only save numpy arrays from 1D to 3D. Use ' \
                                   'Port attributes to save as int, float, or string.'

        # ---- Test data dim of data_dim for new data entry is < 1 or > 3

        out_port = self.create_output_port("new_data")

        data = [1, 2, 4]

        with pytest.raises(ValueError) as error:
            out_port.set_all(data, data_dim=0)

        assert str(error.value) == 'The data dimensions should be 1D, 2D, or 3D.'

        # ---- Test data_dim for new data entry is smaller than actual data

        out_port = self.create_output_port("new_data")

        data = [[1], [2]]

        with pytest.raises(ValueError) as error:
            out_port.set_all(data, data_dim=1)

        assert str(error.value) == 'The dimensions of the data should be equal to or larger ' \
                                   'than the dimensions of the input data.'

        # ---- Test data_dim == 3 and actual size == 1

        out_port = self.create_output_port("new_data")

        data = [1, 2]

        with pytest.raises(ValueError) as error:
            out_port.set_all(data, data_dim=3)

        assert str(error.value) == 'Cannot initialize 1D data in 3D data container.'

    def test_set_all_keep_attributes(self):

        def init_out_port():
            out_port = self.create_output_port("new_data")
            control = self.create_input_port("new_data")

            data = [2, 3, 4]
            out_port.set_all(data)
            out_port.add_attribute(name="test1",
                                   value=1)
            out_port.add_attribute(name="test2",
                                   value=12)
            return out_port, control

        out_port, control = init_out_port()
        out_port.set_all([[]], data_dim=2, keep_attributes=True)

        assert control.get_attribute("test1") == 1
        assert control.get_attribute("test2") == 12
        out_port.del_all_data()

    def test_append_new_data(self):
        # using append even if no data exists
        out_port = self.create_output_port("new_data")
        # ----- 1D input -----
        data = [3, ]
        out_port.append(data)

        control = self.create_input_port("new_data")
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
        out_port = self.create_output_port("new_data")

        # ----- 1D -----
        out_port.append([2, 3, 5], data_dim=1)
        out_port.append([3, 3, 5])

        control = self.create_input_port("new_data")

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
        out_port = self.create_output_port("new_data")

        # Error case (no force)
        out_port.append([2, 3, 5], data_dim=1)

        out_port.append([[[22, 7], [10, 221]],
                         [[223, 46], [1, 15]]],
                        force=True)

        control = self.create_input_port("new_data")

        assert np.array_equal(control.get_all(), [[[22, 7], [10, 221]],
                                                  [[223, 46], [1, 15]]])
        out_port.del_all_data()

    def test_append_existing_data_error(self):
        # ---- port not active ----
        out_port = self.create_output_port("new_data")
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

        with pytest.raises(ValueError) as error:
            out_port.append([[[22, 7], [10, 221]], [[223, 46], [1, 15]]])

        assert str(error.value) == "The port tag 'new_data' is already used with a different " \
                                   "data type. The 'force' parameter can be used to replace " \
                                   "the tag."
        out_port.del_all_data()

    def test_set_data_using_slicing(self):
        out_port = self.create_output_port("new_data")

        out_port.set_all([2, 5, 6, 7, ])
        out_port[3] = 44

        control = self.create_input_port("new_data")

        assert np.array_equal(control.get_all(), [2, 5, 6, 44, ])
        out_port.deactivate()
        out_port[2] = 0
        assert np.array_equal(control.get_all(), [2, 5, 6, 44, ])
        out_port.activate()
        out_port.del_all_data()

    def test_del_all_data(self):
        out_port = self.create_output_port("new_data")
        out_port.set_all([0, 1])
        out_port.del_all_data()

        control = self.create_input_port("new_data")

        with pytest.warns(UserWarning) as warning:
            control.get_all()

        assert len(warning) == 1
        assert warning[0].message.args[0] == "No data under the tag which is linked by the " \
                                             "InputPort."

    def test_add_static_attribute(self):
        out_port = self.create_output_port("new_data")
        out_port.set_all([1])
        out_port.add_attribute("attr1", value=5)
        out_port.add_attribute("attr2", value="no")

        control = self.create_input_port("new_data")
        assert control.get_attribute("attr1") == 5

        out_port.add_attribute("attr1", value=6)
        assert control.get_attribute("attr1") == 6
        assert control.get_attribute("attr2") == "no"

        out_port.deactivate()
        out_port.add_attribute("attr3", value=33)

        with pytest.warns(UserWarning) as warning:
            control.get_attribute("attr3")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The attribute 'attr3' was not found."

        out_port.activate()
        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_add_static_attribute_error(self):
        out_port = self.create_output_port("new_data")

        # add attribute while no data is set
        with pytest.warns(UserWarning) as warning:
            out_port.add_attribute("attr1", value=6)

        # check that only one warning was raised
        assert len(warning) == 1
        # check that the message matches
        assert warning[0].message.args[0] == "Can not store attribute if data tag does not exist."

        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_add_non_static_attribute(self):
        # two different data types
        out_port = self.create_output_port("new_data")
        out_port.set_all([1])
        out_port.add_attribute("attr1", value=[6, 3], static=False)

        control = self.create_input_port("new_data")
        assert np.array_equal(control.get_attribute("attr1"), [6, 3])

        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_append_attribute_data(self):
        out_port = self.create_output_port("new_data")
        out_port.del_all_data()
        out_port.set_all([1])
        out_port.add_attribute("attr1", value=[2, 3], static=False)

        control = self.create_input_port("new_data")
        assert np.array_equal(control.get_attribute("attr1"), [2, 3])

        out_port.append_attribute_data("attr1", value=2)
        assert np.array_equal(control.get_attribute("attr1"), [2, 3, 2])

        out_port.deactivate()
        out_port.append_attribute_data("attr1", value=2)
        assert np.array_equal(control.get_attribute("attr1"), [2, 3, 2])

        out_port.activate()
        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_copy_attributes(self):
        out_port = self.create_output_port("new_data")
        out_port.del_all_attributes()
        out_port.del_all_data()
        out_port.set_all([0, ])

        # some static attributes
        out_port.add_attribute("attr1", 33)
        out_port.add_attribute("attr2", "string")
        out_port.add_attribute("attr3", [1, 2, 3])

        # non static attributes
        out_port.add_attribute("attr_non_static", [3, 4, 5, 6], static=False)

        copy_port = self.create_output_port("other_data")
        copy_port.del_all_attributes()
        copy_port.del_all_data()

        copy_port.set_all([1, ])
        # for attribute overwriting
        copy_port.add_attribute("attr_non_static", [3, 4, 44, 6], static=False)

        control = self.create_input_port("new_data")
        copy_port.copy_attributes(control)

        copy_control = self.create_input_port("other_data")

        assert copy_control.get_attribute("attr1") == 33
        assert copy_control.get_attribute("attr2") == "string"
        assert np.array_equal(copy_control.get_attribute("attr3"), [1, 2, 3])
        assert np.array_equal(copy_control.get_attribute("attr_non_static"), [3, 4, 5, 6])

        copy_port.del_all_attributes()
        copy_port.del_all_data()

        out_port.del_all_attributes()
        out_port.del_all_data()

        port = self.create_output_port("test")
        port.deactivate()
        assert port.copy_attributes(control) is None

        port = self.create_input_port("test")

        with pytest.warns(UserWarning) as warning:
            port.get_all_non_static_attributes()

        assert len(warning) == 1
        assert warning[0].message.args[0] == "No data under the tag which is linked by the " \
                                             "InputPort."

    def test_copy_attributes_same_tag(self):
        out_port1 = self.create_output_port("new_data")
        out_port1.set_all([0, ])

        out_port2 = self.create_output_port("new_data")
        out_port2.set_all([2, ])

        out_port1.add_attribute("attr1", 2)

        control1 = self.create_input_port("new_data")
        out_port2.copy_attributes(control1)

        control2 = self.create_input_port("new_data")
        assert control2.get_attribute("attr1") == 2

        out_port1.del_all_data()
        out_port1.del_all_attributes()

    def test_del_attribute(self):
        out_port = self.create_output_port("new_data")
        out_port.set_all([0, ])

        # static
        out_port.add_attribute("attr1", 4)
        out_port.add_attribute("attr2", 5)

        # non static
        out_port.add_attribute("attr_non_static_1", [1, 2, 3], static=False)
        out_port.add_attribute("attr_non_static_2", [2, 4, 6, 8], static=False)
        out_port.del_attribute("attr1")
        out_port.del_attribute("attr_non_static_1")

        # check is only the chosen attributes are deleted and the rest is still there
        control = self.create_input_port("new_data")

        with pytest.warns(UserWarning) as warning:
            control.get_attribute("attr1")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The attribute 'attr1' was not found."

        assert control.get_attribute("attr2") == 5

        with pytest.warns(UserWarning) as warning:
            control.get_attribute("attr_non_static_1")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The attribute 'attr_non_static_1' was not found."

        assert np.array_equal(control.get_attribute("attr_non_static_2"), [2, 4, 6, 8])

        out_port.del_all_data()
        out_port.del_all_attributes()

    def test_del_attribute_error_case(self):
        out_port = self.create_output_port("new_data")
        out_port.set_all([0, ])

        # deactivated port
        out_port.add_attribute("attr_1", 5.554)
        out_port.deactivate()
        out_port.del_attribute("attr_1")

        control = self.create_input_port("new_data")
        assert control.get_attribute("attr_1") == 5.554

        out_port.activate()
        # not existing
        with pytest.warns(UserWarning) as warning:
            out_port.del_attribute("not_existing")

        # check that only one warning was raised
        assert len(warning) == 1
        # check that the message matches
        assert warning[0].message.args[0] == "Attribute 'not_existing' does not exist and could " \
                                             "not be deleted."

        out_port.del_all_attributes()
        out_port.del_all_data()

    def test_del_all_attributes(self):
        out_port = self.create_output_port("new_data")
        out_port.set_all([0, ])
        out_port.add_attribute("attr_1", 4)
        out_port.add_attribute("attr_2", [1, 3], static=False)
        out_port.del_all_attributes()

        control = self.create_input_port("new_data")

        with pytest.warns(UserWarning) as warning:
            control.get_attribute("attr_1")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The attribute 'attr_1' was not found."

        with pytest.warns(UserWarning) as warning:
            control.get_attribute("attr_2")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The attribute 'attr_2' was not found."

        out_port.del_all_data()

    def test_add_history(self):
        out_port = self.create_output_port("new_data")
        out_port.set_all([0, ])
        out_port.add_history("Test", "history")

        control = self.create_input_port("new_data")
        assert control.get_attribute("History: Test") == "history"

    def test_check_attribute(self):
        out_port = self.create_output_port("new_data")
        out_port.set_all([0, ])
        out_port.add_attribute("static", 5, static=True)
        out_port.add_attribute("non-static", np.arange(1, 11, 1), static=False)

        assert out_port.check_static_attribute("static", 5) == 0
        assert out_port.check_static_attribute("test", 3) == 1
        assert out_port.check_static_attribute("static", 33) == -1

        assert out_port.check_non_static_attribute("non-static", np.arange(1, 11, 1)) == 0
        assert out_port.check_non_static_attribute("test", np.arange(1, 11, 1)) == 1
        assert out_port.check_non_static_attribute("non-static", np.arange(10, 21, 1)) == -1

        out_port.deactivate()

        assert out_port.check_static_attribute("static", 5) is None
        assert out_port.check_non_static_attribute("non-static", np.arange(1, 11, 1)) is None

        out_port.activate()
        out_port.del_all_data()
        out_port.del_all_attributes()
