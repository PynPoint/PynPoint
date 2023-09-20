import os

import pytest
import numpy as np

from pynpoint.core.dataio import DataStorage, InputPort, OutputPort
from pynpoint.util.tests import create_random, create_config, remove_test_data


class TestInputPort:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_random(self.test_dir)
        create_config(self.test_dir+'PynPoint_config.ini')

        file_in = os.path.dirname(__file__) + '/PynPoint_database.hdf5'

        self.storage = DataStorage(file_in)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir)

    def test_create_instance_access_data(self) -> None:

        with pytest.raises(ValueError) as error:
            InputPort('config', self.storage)

        assert str(error.value) == 'The tag name \'config\' is reserved for the central ' \
                                   'configuration of PynPoint.'

        with pytest.raises(ValueError) as error:
            InputPort('fits_header', self.storage)

        assert str(error.value) == 'The tag name \'fits_header\' is reserved for storage of the ' \
                                   'FITS headers.'

        port = InputPort('images', self.storage)

        assert port[0, 0, 0] == pytest.approx(0.00032486907273264834, rel=self.limit, abs=0.)

        data = np.mean(port.get_all())
        assert data == pytest.approx(1.1824138000882435e-05, rel=self.limit, abs=0.)

        assert len(port[0:2, 0, 0]) == 2
        assert port.get_shape() == (5, 11, 11)

        assert port.get_attribute('PIXSCALE') == 0.01
        assert port.get_attribute('PARANG')[0] == 0.

        with pytest.warns(UserWarning):
            assert port.get_attribute('none') is None

    def test_create_instance_access_non_existing_data(self) -> None:

        port = InputPort('test', self.storage)

        with pytest.warns(UserWarning):
            assert port[0, 0, 0] is None

        with pytest.warns(UserWarning):
            assert port.get_all() is None

        with pytest.warns(UserWarning):
            assert port.get_shape() is None

        with pytest.warns(UserWarning):
            assert port.get_attribute('num_files') is None

        with pytest.warns(UserWarning):
            assert port.get_all_non_static_attributes() is None

        with pytest.warns(UserWarning):
            assert port.get_all_static_attributes() is None

    def test_create_instance_no_data_storage(self) -> None:

        port = InputPort('test')

        with pytest.warns(UserWarning):
            assert port[0, 0, 0] is None

        with pytest.warns(UserWarning):
            assert port.get_all() is None

        with pytest.warns(UserWarning):
            assert port.get_shape() is None

        with pytest.warns(UserWarning):
            assert port.get_all_non_static_attributes() is None

        with pytest.warns(UserWarning):
            assert port.get_all_static_attributes() is None

    # def test_get_all_attributes(self) -> None:
    #
    #     port = InputPort('images', self.storage)
    #
    #     assert port.get_all_static_attributes() == {'PIXSCALE': 0.01}
    #     assert port.get_all_non_static_attributes() == ['PARANG', ]
    #
    #     port = OutputPort('images', self.storage)
    #     assert port.del_all_attributes() is None
    #
    #     port = InputPort('images', self.storage)
    #     assert port.get_all_non_static_attributes() is None

    # def test_get_ndim(self) -> None:
    #
    #     with pytest.warns(UserWarning) as warning:
    #         ndim = InputPort('images', None).get_ndim()
    #
    #     assert len(warning) == 1
    #
    #     assert warning[0].message.args[0] == 'InputPort can not load data ' \
    #                                          'unless a database is connected.'
    #
    #     assert ndim is None
    #
    #     port = InputPort('images', self.storage)
    #     assert port.get_ndim() == 3
