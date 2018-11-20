import os
import warnings

import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.BackgroundSubtraction import MeanBackgroundSubtractionModule, \
                                                             SimpleBackgroundSubtractionModule, \
                                                             NoddingBackgroundModule, \
                                                             DitheringBackgroundModule
from PynPoint.ProcessingModules.StackingAndSubsampling import MeanCubeModule
from PynPoint.Util.TestTools import create_config, create_fake, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestBackgroundSubtraction(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_fake(path=self.test_dir+"dither",
                    ndit=[20, 20, 20, 20],
                    nframes=[20, 20, 20, 20],
                    exp_no=[1, 2, 3, 4],
                    npix=(100, 100),
                    fwhm=3.,
                    x0=[25, 75, 75, 25],
                    y0=[75, 75, 25, 25],
                    angles=[[0., 25.], [25., 50.], [50., 75.], [75., 100.]],
                    sep=None,
                    contrast=None)

        create_fake(path=self.test_dir+"star",
                    ndit=[10, 10, 10, 10],
                    nframes=[10, 10, 10, 10],
                    exp_no=[1, 3, 5, 7],
                    npix=(100, 100),
                    fwhm=3.,
                    x0=[50, 50, 50, 50],
                    y0=[50, 50, 50, 50],
                    angles=[[0., 25.], [25., 50.], [50., 75.], [75., 100.]],
                    sep=None,
                    contrast=None)

        create_fake(path=self.test_dir+"sky",
                    ndit=[5, 5, 5, 5],
                    nframes=[5, 5, 5, 5],
                    exp_no=[2, 4, 6, 8],
                    npix=(100, 100),
                    fwhm=None,
                    x0=[50, 50, 50, 50],
                    y0=[50, 50, 50, 50],
                    angles=[[0., 25.], [25., 50.], [50., 75.], [75., 100.]],
                    sep=None,
                    contrast=None)

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["dither", "star", "sky"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read1",
                                 image_tag="dither",
                                 input_dir=self.test_dir+"dither")

        self.pipeline.add_module(read)

        read = FitsReadingModule(name_in="read2",
                                 image_tag="star",
                                 input_dir=self.test_dir+"star")

        self.pipeline.add_module(read)

        read = FitsReadingModule(name_in="read3",
                                 image_tag="sky",
                                 input_dir=self.test_dir+"sky")

        self.pipeline.add_module(read)

        self.pipeline.run_module("read1")
        self.pipeline.run_module("read2")
        self.pipeline.run_module("read3")

        data = self.pipeline.get_data("dither")
        assert np.allclose(data[0, 74, 24], 0.05304008435511765, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010033896953157959, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

        data = self.pipeline.get_data("star")
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        data = self.pipeline.get_data("sky")
        assert np.allclose(data[0, 50, 50], -7.613171257478652e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 8.937360237872607e-07, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_simple_background(self):

        simple = SimpleBackgroundSubtractionModule(shift=20,
                                                   name_in="simple",
                                                   image_in_tag="dither",
                                                   image_out_tag="simple")

        self.pipeline.add_module(simple)
        self.pipeline.run_module("simple")

        data = self.pipeline.get_data("simple")
        assert np.allclose(data[0, 74, 74], -0.05288064325101517, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.7755575615628916e-22, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

    def test_mean_background_shift(self):

        mean = MeanBackgroundSubtractionModule(shift=20,
                                               cubes=1,
                                               name_in="mean2",
                                               image_in_tag="dither",
                                               image_out_tag="mean2")

        self.pipeline.add_module(mean)
        self.pipeline.run_module("mean2")

        data = self.pipeline.get_data("mean2")
        assert np.allclose(data[0, 74, 24], 0.0530465391626132, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.3970872216676808e-07, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

    def test_mean_background_nframes(self):

        mean = MeanBackgroundSubtractionModule(shift=None,
                                               cubes=1,
                                               name_in="mean1",
                                               image_in_tag="dither",
                                               image_out_tag="mean1")

        self.pipeline.add_module(mean)
        self.pipeline.run_module("mean1")

        data = self.pipeline.get_data("mean1")
        assert np.allclose(data[0, 74, 24], 0.0530465391626132, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.3970872216676808e-07, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

    def test_dithering_attributes(self):

        pca_dither = DitheringBackgroundModule(name_in="pca_dither1",
                                               image_in_tag="dither",
                                               image_out_tag="pca_dither1",
                                               center=None,
                                               cubes=None,
                                               size=0.8,
                                               gaussian=0.1,
                                               subframe=0.5,
                                               pca_number=5,
                                               mask_star=0.1,
                                               mask_planet=None,
                                               crop=True,
                                               prepare=True,
                                               pca_background=True,
                                               combine="pca")

        self.pipeline.add_module(pca_dither)
        self.pipeline.run_module("pca_dither1")

        data = self.pipeline.get_data("dither_dither_crop1")
        assert np.allclose(data[0, 14, 14], 0.05304008435511765, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.0002606205855710527, rtol=1e-6, atol=0.)
        assert data.shape == (80, 31, 31)

        data = self.pipeline.get_data("dither_dither_star1")
        assert np.allclose(data[0, 14, 14], 0.05304008435511765, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.0010414302265833978, rtol=1e-6, atol=0.)
        assert data.shape == (20, 31, 31)

        data = self.pipeline.get_data("dither_dither_mean1")
        assert np.allclose(data[0, 14, 14], 0.0530465391626132, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.0010426228104479674, rtol=1e-6, atol=0.)
        assert data.shape == (20, 31, 31)

        data = self.pipeline.get_data("dither_dither_background1")
        assert np.allclose(data[0, 14, 14], -0.00010629310882411674, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 3.5070523360436835e-07, rtol=1e-6, atol=0.)
        assert data.shape == (60, 31, 31)

        data = self.pipeline.get_data("dither_dither_pca_fit1")
        assert np.allclose(data[0, 14, 14], 1.5196412298279846e-05, rtol=1e-5, atol=0.)
        assert np.allclose(np.mean(data), 1.9779802529804516e-07, rtol=1e-4, atol=0.)
        assert data.shape == (20, 31, 31)

        data = self.pipeline.get_data("dither_dither_pca_res1")
        assert np.allclose(data[0, 14, 14], 0.05302488794281937, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.0010412324285580998, rtol=1e-6, atol=0.)
        assert data.shape == (20, 31, 31)

        data = self.pipeline.get_data("dither_dither_pca_mask1")
        assert np.allclose(data[0, 14, 14], 0., rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.9531737773152965, rtol=1e-6, atol=0.)
        assert data.shape == (20, 31, 31)

        data = self.pipeline.get_data("pca_dither1")
        assert np.allclose(data[0, 14, 14], 0.05302488794281937, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.001040627977720779, rtol=1e-6, atol=0.)
        assert data.shape == (80, 31, 31)

        data = self.pipeline.get_attribute("dither_dither_pca_res1", "STAR_POSITION", static=False)
        assert np.allclose(data[0, 0], [15., 15.], rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 15., rtol=1e-6, atol=0.)
        assert data.shape == (20, 2)

    def test_dithering_center(self):

        pca_dither = DitheringBackgroundModule(name_in="pca_dither2",
                                               image_in_tag="dither",
                                               image_out_tag="pca_dither2",
                                               center=((25., 75.),
                                                       (75., 75.),
                                                       (75., 25.),
                                                       (25., 25.)),
                                               cubes=1,
                                               size=0.8,
                                               gaussian=0.1,
                                               subframe=None,
                                               pca_number=5,
                                               mask_star=0.1,
                                               mask_planet=None,
                                               bad_pixel=None,
                                               crop=True,
                                               prepare=True,
                                               pca_background=True,
                                               combine="pca")

        self.pipeline.add_module(pca_dither)
        self.pipeline.run_module("pca_dither2")

        data = self.pipeline.get_data("pca_dither2")
        assert np.allclose(data[0, 14, 14], 0.05302488794328089, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.0010406279782666378, rtol=1e-3, atol=0.)
        assert data.shape == (80, 31, 31)

    def test_nodding_background(self):

        mean = MeanCubeModule(name_in="mean",
                              image_in_tag="sky",
                              image_out_tag="mean")

        self.pipeline.add_module(mean)
        self.pipeline.run_module("mean")

        data = self.pipeline.get_data("mean")
        assert np.allclose(data[0, 50, 50], 1.270877476321969e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 8.937360237872607e-07, rtol=limit, atol=0.)
        assert data.shape == (4, 100, 100)

        attribute = self.pipeline.get_attribute("mean", "INDEX", static=False)
        assert np.allclose(np.mean(attribute), 1.5, rtol=limit, atol=0.)
        assert attribute.shape == (4, )

        attribute = self.pipeline.get_attribute("mean", "NFRAMES", static=False)
        assert np.allclose(np.mean(attribute), 1, rtol=limit, atol=0.)
        assert attribute.shape == (4, )

        nodding = NoddingBackgroundModule(name_in="nodding",
                                          sky_in_tag="mean",
                                          science_in_tag="star",
                                          image_out_tag="nodding",
                                          mode="both")

        self.pipeline.add_module(nodding)
        self.pipeline.run_module("nodding")

        data = self.pipeline.get_data("nodding")
        assert np.allclose(data[0, 50, 50], 0.09797142624717381, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.945087327935862e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)
