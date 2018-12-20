import os
import warnings

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.psfpreparation import PSFpreparationModule, AngleInterpolationModule, \
                                               AngleCalculationModule, SDIpreparationModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestPSFpreparation(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"prep")
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["prep"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read",
                                 input_dir=self.test_dir+"prep")

        self.pipeline.add_module(read)
        self.pipeline.run_module("read")

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_angle_interpolation(self):

        angle = AngleInterpolationModule(name_in="angle1",
                                         data_tag="read")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle1")

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.allclose(data[0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[15], 7.777777777777778, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_angle_calculation(self):

        self.pipeline.set_attribute("read", "LATITUDE", -25.)
        self.pipeline.set_attribute("read", "LONGITUDE", -70.)
        self.pipeline.set_attribute("read", "DIT", 1.)

        self.pipeline.set_attribute("read", "RA", (90., 90., 90., 90.), static=False)
        self.pipeline.set_attribute("read", "DEC", (-51., -51., -51., -51.), static=False)
        self.pipeline.set_attribute("read", "PUPIL", (90., 90., 90., 90.), static=False)

        date = ("2012-12-01T07:09:00.0000", "2012-12-01T07:09:01.0000", \
                "2012-12-01T07:09:02.0000", "2012-12-01T07:09:03.0000")

        self.pipeline.set_attribute("read", "DATE", date, static=False)

        angle = AngleCalculationModule(instrument="NACO",
                                       name_in="angle2",
                                       data_tag="read")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle2")

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.allclose(data[0], -55.0432738327744, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -55.000752378269965, rtol=limit, atol=0.)
        assert data.shape == (40, )

        angle = AngleCalculationModule(instrument="SPHERE/IRDIS",
                                       name_in="angle3",
                                       data_tag="read")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle3")

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.allclose(data[0], 170.38885107983538, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 170.46124323291738, rtol=limit, atol=0.)
        assert data.shape == (40, )

        angle = AngleCalculationModule(instrument="SPHERE/IFS",
                                       name_in="angle4",
                                       data_tag="read")

        self.pipeline.add_module(angle)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("angle4")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "AngleCalculationModule has not been tested for " \
                                             "SPHERE/IFS data."

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.allclose(data[0], 270.8688510798354, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 270.9702735149575, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_angle_interpolation_mismatch(self):

        self.pipeline.set_attribute("read", "NDIT", [9, 9, 9, 9], static=False)

        angle = AngleInterpolationModule(name_in="angle5",
                                         data_tag="read")

        self.pipeline.add_module(angle)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("angle5")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "There is a mismatch between the NDIT and NFRAMES " \
                                             "values. The derotation angles are calculated with " \
                                             "a linear interpolation by using NFRAMES steps. A " \
                                             "frame selection should be applied after the " \
                                             "derotation angles are calculated."

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.allclose(data[0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[15], 7.777777777777778, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_psf_preparation_norm_resize_mask(self):

        prep = PSFpreparationModule(name_in="prep1",
                                    image_in_tag="read",
                                    image_out_tag="prep1",
                                    mask_out_tag="mask1",
                                    norm=True,
                                    resize=2.,
                                    cent_size=0.1,
                                    edge_size=1.0)

        self.pipeline.add_module(prep)
        self.pipeline.run_module("prep1")

        data = self.pipeline.get_data("prep1")
        assert np.allclose(data[0, 0, 0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[0, 25, 25], 0., rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], 0., rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0001818623671899089, rtol=limit, atol=0.)
        assert data.shape == (40, 200, 200)

        data = self.pipeline.get_data("mask1")
        assert np.allclose(data[0, 0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[120, 120], 1., rtol=limit, atol=0.)
        assert np.allclose(data[100, 100], 0., rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.1067, rtol=limit, atol=0.)
        assert data.shape == (200, 200)

    def test_psf_preparation_none(self):

        prep = PSFpreparationModule(name_in="prep2",
                                    image_in_tag="read",
                                    image_out_tag="prep2",
                                    mask_out_tag="mask2",
                                    norm=False,
                                    resize=None,
                                    cent_size=None,
                                    edge_size=None)

        self.pipeline.add_module(prep)
        self.pipeline.run_module("prep2")

        data = self.pipeline.get_data("prep2")
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], -0.000287573978535779, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_psf_preparation_no_mask_out(self):

        prep = PSFpreparationModule(name_in="prep3",
                                    image_in_tag="read",
                                    image_out_tag="prep3",
                                    mask_out_tag=None,
                                    norm=False,
                                    resize=None,
                                    cent_size=None,
                                    edge_size=None)

        self.pipeline.add_module(prep)
        self.pipeline.run_module("prep3")

        data = self.pipeline.get_data("prep3")
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], -0.000287573978535779, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_sdi_preparation(self):

        sdi = SDIpreparationModule(name_in="sdi",
                                   wavelength=(0.65, 0.6),
                                   width=(0.1, 0.5),
                                   image_in_tag="read",
                                   image_out_tag="sdi")

        self.pipeline.add_module(sdi)
        self.pipeline.run_module("sdi")

        data = self.pipeline.get_data("sdi")
        assert np.allclose(data[0, 25, 25], -2.6648118007008814e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.0042892634995876e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        attribute = self.pipeline.get_attribute("sdi", "History: Wavelength center")
        assert attribute == "(line, continuum) = (0.65, 0.6)"

        attribute = self.pipeline.get_attribute("sdi", "History: Wavelength width")
        assert attribute == "(line, continuum) = (0.1, 0.5)"
