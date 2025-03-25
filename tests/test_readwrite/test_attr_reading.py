import os

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.attr_reading import (
    ParangReadingModule,
    AttributeReadingModule,
    WavelengthReadingModule,
)
from pynpoint.util.tests import create_config, create_random, remove_test_data


class TestAttributeReading:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + "/"

        create_random(self.test_dir, nimages=10)
        create_config(self.test_dir + "PynPoint_config.ini")

        np.savetxt(self.test_dir + "parang.dat", np.arange(10.0, 20.0, 1.0))
        np.savetxt(self.test_dir + "new.dat", np.arange(20.0, 30.0, 1.0))
        np.savetxt(self.test_dir + "attribute.dat", np.arange(0, 10, 1), fmt="%i")
        np.savetxt(self.test_dir + "wavelength.dat", np.arange(0.0, 10.0, 1.0))

        data2d = np.random.normal(loc=0, scale=2e-4, size=(10, 10))
        np.savetxt(self.test_dir + "data_2d.dat", data2d)

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(
            self.test_dir,
            files=[
                "parang.dat",
                "new.dat",
                "attribute.dat",
                "data_2d.dat",
                "wavelength.dat",
            ],
        )

    def test_input_data(self) -> None:

        data = self.pipeline.get_data("images")
        assert np.sum(data) == pytest.approx(
            0.007133341144768919, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

    def test_parang_reading(self) -> None:

        module = ParangReadingModule(
            name_in="parang1",
            data_tag="images",
            file_name="parang.dat",
            input_dir=None,
            overwrite=True,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("parang1")

        data = self.pipeline.get_data("header_images/PARANG")
        assert data.dtype == "float64"
        assert data == pytest.approx(
            np.arange(10.0, 20.0, 1.0), rel=self.limit, abs=0.0
        )
        assert data.shape == (10,)

    def test_parang_reading_same(self) -> None:

        module = ParangReadingModule(
            name_in="parang2",
            data_tag="images",
            file_name="parang.dat",
            input_dir=None,
            overwrite=True,
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("parang2")

        assert len(warning) == 1
        assert (
            warning[0].message.args[0] == "The PARANG attribute is already present and "
            "contains the same values as are present in "
            "parang.dat."
        )

    def test_parang_reading_present(self) -> None:

        module = ParangReadingModule(
            name_in="parang3",
            data_tag="images",
            file_name="new.dat",
            input_dir=None,
            overwrite=False,
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("parang3")

        assert len(warning) == 1
        assert (
            warning[0].message.args[0]
            == "The PARANG attribute is already present. Set the "
            "'overwrite' parameter to True in order to "
            "overwrite the values with new.dat."
        )

    def test_parang_reading_overwrite(self) -> None:

        module = ParangReadingModule(
            file_name="new.dat",
            name_in="parang4",
            input_dir=None,
            data_tag="images",
            overwrite=True,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("parang4")

    def test_parang_reading_2d(self) -> None:

        module = ParangReadingModule(
            name_in="parang6",
            data_tag="images",
            file_name="data_2d.dat",
            input_dir=None,
            overwrite=False,
        )

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("parang6")

        assert (
            str(error.value)
            == "The input file data_2d.dat should contain a 1D data set with "
            "the parallactic angles."
        )

    def test_attribute_reading(self) -> None:

        module = AttributeReadingModule(
            file_name="attribute.dat",
            attribute="EXP_NO",
            name_in="attribute1",
            input_dir=None,
            data_tag="images",
            overwrite=False,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("attribute1")

        data = self.pipeline.get_data("header_images/EXP_NO")
        assert data.dtype == "int64"
        assert data == pytest.approx(np.arange(10), rel=self.limit, abs=0.0)
        assert data.shape == (10,)

    def test_attribute_reading_present(self) -> None:

        module = AttributeReadingModule(
            file_name="parang.dat",
            attribute="PARANG",
            name_in="attribute3",
            input_dir=None,
            data_tag="images",
            overwrite=False,
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("attribute3")

        assert (
            warning[0].message.args[0]
            == "The attribute 'PARANG' is already present. Set "
            "the 'overwrite' parameter to True in order to "
            "overwrite the values with parang.dat."
        )

    def test_attribute_reading_invalid(self) -> None:

        module = AttributeReadingModule(
            file_name="attribute.dat",
            attribute="test",
            name_in="attribute4",
            input_dir=None,
            data_tag="images",
            overwrite=False,
        )

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("attribute4")

        assert str(error.value) == "'test' is not a valid attribute."

    def test_attribute_reading_2d(self) -> None:

        module = AttributeReadingModule(
            file_name="data_2d.dat",
            attribute="DITHER_X",
            name_in="attribute5",
            input_dir=None,
            data_tag="images",
            overwrite=False,
        )

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("attribute5")

        assert (
            str(error.value)
            == "The input file data_2d.dat should contain a 1D list with "
            "attributes."
        )

    def test_attribute_reading_same(self) -> None:

        module = AttributeReadingModule(
            file_name="attribute.dat",
            attribute="EXP_NO",
            name_in="attribute6",
            input_dir=None,
            data_tag="images",
            overwrite=True,
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("attribute6")

        assert len(warning) == 1
        assert (
            warning[0].message.args[0]
            == "The 'EXP_NO' attribute is already present and "
            "contains the same values as are present in "
            "attribute.dat."
        )

    def test_attribute_reading_overwrite(self) -> None:

        module = AttributeReadingModule(
            file_name="parang.dat",
            attribute="PARANG",
            name_in="attribute7",
            input_dir=None,
            data_tag="images",
            overwrite=True,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("attribute7")

        data = self.pipeline.get_attribute("images", "PARANG", static=False)
        assert data == pytest.approx(
            np.arange(10.0, 20.0, 1.0), rel=self.limit, abs=0.0
        )

    def test_wavelength_reading(self) -> None:

        module = WavelengthReadingModule(
            file_name="wavelength.dat",
            name_in="wavelength1",
            input_dir=None,
            data_tag="images",
            overwrite=False,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("wavelength1")

        data = self.pipeline.get_data("header_images/WAVELENGTH")
        assert data.dtype == "float64"
        assert data == pytest.approx(np.arange(10.0), rel=self.limit, abs=0.0)
        assert data.shape == (10,)

    def test_wavelength_reading_same(self) -> None:

        module = WavelengthReadingModule(
            file_name="wavelength.dat",
            name_in="wavelength2",
            input_dir=None,
            data_tag="images",
            overwrite=True,
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("wavelength2")

        assert len(warning) == 1
        assert (
            warning[0].message.args[0]
            == "The WAVELENGTH attribute is already present and "
            "contains the same values as are present in "
            "wavelength.dat."
        )

    def test_wavelength_reading_present(self) -> None:

        module = WavelengthReadingModule(
            file_name="new.dat",
            name_in="wavelength3",
            input_dir=None,
            data_tag="images",
            overwrite=False,
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("wavelength3")

        assert len(warning) == 1
        assert (
            warning[0].message.args[0]
            == "The WAVELENGTH attribute is already present. Set "
            "the 'overwrite' parameter to True in order to "
            "overwrite the values with new.dat."
        )

    def test_wavelength_reading_overwrite(self) -> None:

        module = WavelengthReadingModule(
            file_name="new.dat",
            name_in="wavelength4",
            input_dir=None,
            data_tag="images",
            overwrite=True,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("wavelength4")

    def test_wavelength_reading_2d(self) -> None:

        module = WavelengthReadingModule(
            file_name="data_2d.dat",
            name_in="wavelength6",
            input_dir=None,
            data_tag="images",
            overwrite=False,
        )

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("wavelength6")

        assert (
            str(error.value)
            == "The input file data_2d.dat should contain a 1D data set with "
            "the wavelengths."
        )
