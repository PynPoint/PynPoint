from __future__ import absolute_import

import os
import warnings

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.core.dataio import ConfigPort, DataStorage
from pynpoint.util.tests import create_config, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestConfigPort(object):

    def setup_class(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        create_config(self.test_dir+"PynPoint_config.ini")

    def teardown_class(self):
        remove_test_data(self.test_dir)

    def test_create_config_port(self):
        storage = DataStorage(self.test_dir + "PynPoint_database.hdf5")

        with pytest.raises(ValueError) as error:
            ConfigPort("images", storage)

        assert str(error.value) == "The tag name of the central configuration should be 'config'."

        port = ConfigPort("config", None)

        with pytest.warns(UserWarning) as warning:
            check_error = port._check_error_cases()

        assert len(warning) == 1
        assert warning[0].message.args[0] == "ConfigPort can not load data unless a database is " \
                                             "connected."

        assert not check_error

        port = ConfigPort("config", storage)
        assert isinstance(port, ConfigPort)

        with pytest.warns(UserWarning) as warning:
            port._check_error_cases()

        assert len(warning) == 1
        assert warning[0].message.args[0] == "No data under the tag which is linked by the ConfigPort."

    def test_get_config_attribute(self):
        create_config(self.test_dir+"PynPoint_config.ini")
        Pypeline(self.test_dir, self.test_dir, self.test_dir)

        storage = DataStorage(self.test_dir + "PynPoint_database.hdf5")
        port = ConfigPort("config", None)

        with pytest.warns(UserWarning) as warning:
            attribute = port.get_attribute("CPU")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "ConfigPort can not load data unless a database is " \
                                             "connected."

        assert attribute is None

        port = ConfigPort("config", storage)

        attribute = port.get_attribute("CPU")
        assert attribute == 1

        attribute = port.get_attribute("NFRAMES")
        assert attribute == "NAXIS3"

        attribute = port.get_attribute("PIXSCALE")
        assert np.allclose(attribute, 0.027, rtol=limit, atol=0.)

        with pytest.warns(UserWarning) as warning:
            attribute = port.get_attribute("test")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "No attribute found - requested: test."

        assert attribute is None
