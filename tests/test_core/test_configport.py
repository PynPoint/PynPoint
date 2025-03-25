import os

import pytest

from pynpoint.core.pypeline import Pypeline
from pynpoint.core.dataio import ConfigPort, DataStorage
from pynpoint.util.tests import create_config, remove_test_data


class TestConfigPort:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + "/"
        create_config(self.test_dir + "PynPoint_config.ini")

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir)

    def test_create_config_port(self) -> None:

        storage = DataStorage(self.test_dir + "PynPoint_database.hdf5")

        with pytest.raises(ValueError) as error:
            ConfigPort("images", storage)

        assert (
            str(error.value) == "The tag name of the central configuration should be "
            "'config'."
        )

        port = ConfigPort("config", None)

        with pytest.warns(UserWarning) as warning:
            check_error = port._check_error_cases()

        assert len(warning) == 1

        assert (
            warning[0].message.args[0]
            == "ConfigPort can not load data unless a database is "
            "connected."
        )

        assert not check_error

        port = ConfigPort("config", storage)
        assert isinstance(port, ConfigPort)

        with pytest.warns(UserWarning) as warning:
            port._check_error_cases()

        assert len(warning) == 1

        assert (
            warning[0].message.args[0]
            == "No data under the tag which is linked by the "
            "ConfigPort."
        )

    def test_get_config_attribute(self) -> None:

        create_config(self.test_dir + "PynPoint_config.ini")
        Pypeline(self.test_dir, self.test_dir, self.test_dir)

        storage = DataStorage(self.test_dir + "PynPoint_database.hdf5")
        port = ConfigPort("config", None)

        with pytest.warns(UserWarning) as warning:
            attribute = port.get_attribute("CPU")

        assert len(warning) == 1

        assert (
            warning[0].message.args[0]
            == "ConfigPort can not load data unless a database is "
            "connected."
        )

        assert attribute is None

        port = ConfigPort("config", storage)

        attribute = port.get_attribute("CPU")
        assert attribute == 1

        attribute = port.get_attribute("NFRAMES")
        assert attribute == "NAXIS3"

        attribute = port.get_attribute("PIXSCALE")
        assert attribute == pytest.approx(0.027, rel=self.limit, abs=0.0)

        with pytest.warns(UserWarning) as warning:
            attribute = port.get_attribute("test")

        assert len(warning) == 1

        assert warning[0].message.args[0] == "The attribute 'test' was not found."

        assert attribute is None
