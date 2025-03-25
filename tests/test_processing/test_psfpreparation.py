import os

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.psfpreparation import (
    PSFpreparationModule,
    AngleInterpolationModule,
    AngleCalculationModule,
    SDIpreparationModule,
    SortParangModule,
)
from pynpoint.util.tests import (
    create_config,
    create_star_data,
    create_ifs_data,
    remove_test_data,
)


class TestPsfPreparation:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(self.test_dir + "prep")
        create_ifs_data(self.test_dir + "prep_ifs")
        create_config(self.test_dir + "PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=["prep", "prep_ifs"])

    def test_read_data(self) -> None:

        module = FitsReadingModule(
            name_in="read", image_tag="read", input_dir=self.test_dir + "prep"
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("read")

        data = self.pipeline.get_data("read")
        assert np.sum(data) == pytest.approx(
            105.54278879805277, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

        module = FitsReadingModule(
            name_in="read_ifs",
            image_tag="read_ifs",
            input_dir=self.test_dir + "prep_ifs",
            ifs_data=True,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("read_ifs")

        data = self.pipeline.get_data("read_ifs")
        assert np.sum(data) == pytest.approx(749.8396528807369, rel=self.limit, abs=0.0)
        assert data.shape == (3, 10, 21, 21)

    def test_angle_interpolation(self) -> None:

        module = AngleInterpolationModule(name_in="angle1", data_tag="read")

        self.pipeline.add_module(module)
        self.pipeline.run_module("angle1")

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.sum(data) == pytest.approx(900.0, rel=self.limit, abs=0.0)
        assert data.shape == (10,)

    def test_angle_calculation(self) -> None:

        self.pipeline.set_attribute("read", "LATITUDE", -25.0)
        self.pipeline.set_attribute("read", "LONGITUDE", -70.0)
        self.pipeline.set_attribute("read", "DIT", 1.0)

        self.pipeline.set_attribute(
            "read", "RA", (90.0, 90.0, 90.0, 90.0), static=False
        )
        self.pipeline.set_attribute(
            "read", "DEC", (-51.0, -51.0, -51.0, -51.0), static=False
        )
        self.pipeline.set_attribute(
            "read", "PUPIL", (90.0, 90.0, 90.0, 90.0), static=False
        )

        date = (
            "2012-12-01T07:09:00.0000",
            "2012-12-01T07:09:01.0000",
            "2012-12-01T07:09:02.0000",
            "2012-12-01T07:09:03.0000",
        )

        self.pipeline.set_attribute("read", "DATE", date, static=False)

        module = AngleCalculationModule(
            instrument="NACO", name_in="angle2", data_tag="read"
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("angle2")

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.sum(data) == pytest.approx(
            -550.2338293743467, rel=self.limit, abs=0.0
        )
        assert data.shape == (10,)

        self.pipeline.set_attribute(
            "read", "RA", (60000.0, 60000.0, 60000.0, 60000.0), static=False
        )

        self.pipeline.set_attribute(
            "read", "DEC", (-510000.0, -510000.0, -510000.0, -510000.0), static=False
        )

        module = AngleCalculationModule(
            instrument="SPHERE/IRDIS", name_in="angle3", data_tag="read"
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("angle3")

        warning_0 = (
            "For SPHERE data it is recommended to use the header keyword 'ESO INS4 "
            "DROT2 RA' to specify the object's right ascension. The input will be "
            "parsed accordingly. Using the regular 'RA' keyword will lead to wrong "
            "parallactic angles."
        )

        warning_1 = (
            "For SPHERE data it is recommended to use the header keyword 'ESO INS4 "
            "DROT2 DEC' to specify the object's declination. The input will be parsed "
            "accordingly. Using the regular 'DEC' keyword will lead to wrong "
            "parallactic angles."
        )

        if len(warning) == 2:
            assert warning[0].message.args[0] == warning_0
            assert warning[1].message.args[0] == warning_1

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.sum(data) == pytest.approx(
            1704.2202367435675, rel=self.limit, abs=0.0
        )
        assert data.shape == (10,)

        module = AngleCalculationModule(
            instrument="SPHERE/IFS", name_in="angle4", data_tag="read"
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("angle4")

        warning_0 = "AngleCalculationModule has not been tested for SPHERE/IFS data."

        warning_1 = (
            "For SPHERE data it is recommended to use the header keyword 'ESO INS4 "
            "DROT2 RA' to specify the object's right ascension. The input will be "
            "parsed accordingly. Using the regular 'RA' keyword will lead to wrong "
            "parallactic angles."
        )

        warning_2 = (
            "For SPHERE data it is recommended to use the header keyword 'ESO INS4 "
            "DROT2 DEC' to specify the object's declination. The input will be parsed "
            "accordingly. Using the regular 'DEC' keyword will lead to wrong "
            "parallactic angles."
        )

        if len(warning) == 3:
            assert warning[0].message.args[0] == warning_0
            assert warning[1].message.args[0] == warning_1
            assert warning[2].message.args[0] == warning_2

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.sum(data) == pytest.approx(
            -890.8506514377593, rel=self.limit, abs=0.0
        )
        assert data.shape == (10,)

    def test_angle_sort(self) -> None:

        index = self.pipeline.get_data("header_read/INDEX")
        self.pipeline.set_attribute("read", "INDEX", index[::-1], static=False)

        module = SortParangModule(
            name_in="sort1", image_in_tag="read", image_out_tag="read_sorted"
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("sort1")
        self.pipeline.set_attribute("read", "INDEX", index, static=False)

        parang = self.pipeline.get_data("header_read/PARANG")[::-1]
        parang_sort = self.pipeline.get_data("header_read_sorted/PARANG")
        assert np.sum(parang) == pytest.approx(
            np.sum(parang_sort), rel=self.limit, abs=0.0
        )

        parang_set = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        self.pipeline.set_attribute("read_ifs", "PARANG", parang_set, static=False)

        data = self.pipeline.get_data("read_sorted")
        assert np.sum(data[0]) == pytest.approx(
            9.71156815235485, rel=self.limit, abs=0.0
        )

    def test_angle_sort_ifs(self) -> None:

        index = self.pipeline.get_data("header_read_ifs/INDEX")
        self.pipeline.set_attribute("read_ifs", "INDEX", index[::-1], static=False)

        module = SortParangModule(
            name_in="sort2", image_in_tag="read_ifs", image_out_tag="read_ifs_sorted"
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("sort2")
        self.pipeline.set_attribute("read_ifs", "INDEX", index, static=False)

        parang = self.pipeline.get_data("header_read_ifs/PARANG")[::-1]
        parang_sort = self.pipeline.get_data("header_read_ifs_sorted/PARANG")
        assert np.sum(parang) == pytest.approx(
            np.sum(parang_sort), rel=self.limit, abs=0.0
        )

        data = self.pipeline.get_data("read_ifs_sorted")
        assert np.sum(data[0, 0]) == pytest.approx(
            21.185139976163477, rel=self.limit, abs=0.0
        )

    def test_angle_interpolation_mismatch(self) -> None:

        self.pipeline.set_attribute("read", "NDIT", [9, 9, 9, 9], static=False)

        module = AngleInterpolationModule(name_in="angle5", data_tag="read")

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("angle5")

        warning_0 = (
            "There is a mismatch between the NDIT and NFRAMES values. The parallactic "
            "angles are calculated with a linear interpolation by using NFRAMES steps. "
            "A frame selection should be applied after the parallactic angles are "
            "calculated."
        )

        if len(warning) == 1:
            assert warning[0].message.args[0] == warning_0

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.sum(data) == pytest.approx(900.0, rel=self.limit, abs=0.0)
        assert data.shape == (10,)

    def test_psf_preparation_norm_mask(self) -> None:

        module = PSFpreparationModule(
            name_in="prep1",
            image_in_tag="read",
            image_out_tag="prep1",
            mask_out_tag="mask1",
            norm=True,
            cent_size=0.1,
            edge_size=1.0,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("prep1")

        data = self.pipeline.get_data("prep1")
        assert np.sum(data) == pytest.approx(
            -1.5844830188044685, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

        data = self.pipeline.get_data("mask1")
        assert np.sum(data) == pytest.approx(52, rel=self.limit, abs=0.0)
        assert data.shape == (11, 11)

    def test_psf_preparation_none(self) -> None:

        module = PSFpreparationModule(
            name_in="prep2",
            image_in_tag="read",
            image_out_tag="prep2",
            mask_out_tag="mask2",
            norm=False,
            cent_size=None,
            edge_size=None,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("prep2")

        data = self.pipeline.get_data("prep2")
        assert np.sum(data) == pytest.approx(
            105.54278879805277, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

    def test_psf_preparation_no_mask_out(self) -> None:

        module = PSFpreparationModule(
            name_in="prep3",
            image_in_tag="read",
            image_out_tag="prep3",
            mask_out_tag=None,
            norm=False,
            cent_size=None,
            edge_size=None,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("prep3")

        data = self.pipeline.get_data("prep3")
        assert np.sum(data) == pytest.approx(
            105.54278879805277, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

    def test_psf_preparation_sdi(self) -> None:

        module = PSFpreparationModule(
            name_in="prep4",
            image_in_tag="read_ifs",
            image_out_tag="prep4",
            mask_out_tag=None,
            norm=False,
            cent_size=None,
            edge_size=None,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("prep4")

        data = self.pipeline.get_data("prep4")
        assert np.sum(data) == pytest.approx(749.8396528807369, rel=self.limit, abs=0.0)
        assert data.shape == (3, 10, 21, 21)

    def test_sdi_preparation(self) -> None:

        module = SDIpreparationModule(
            name_in="sdi",
            wavelength=(0.65, 0.6),
            width=(0.1, 0.5),
            image_in_tag="read",
            image_out_tag="sdi",
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("sdi")

        data = self.pipeline.get_data("sdi")
        assert np.sum(data) == pytest.approx(
            21.084666133914183, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

        attribute = self.pipeline.get_attribute("sdi", "History: SDIpreparationModule")
        assert attribute == "(line, continuum) = (0.65, 0.6)"
