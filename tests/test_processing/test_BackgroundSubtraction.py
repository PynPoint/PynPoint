import os
import warnings

import h5py
import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.Core.DataIO import DataStorage
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.BackgroundSubtraction import MeanBackgroundSubtractionModule, SimpleBackgroundSubtractionModule, \
                                                             PCABackgroundPreparationModule, PCABackgroundSubtractionModule, \
                                                             DitheringBackgroundModule, NoddingBackgroundModule
from PynPoint.Util.TestTools import create_config, create_fake

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    test_dir = os.path.dirname(__file__)+"/"

    os.makedirs(test_dir+"dither")
    os.makedirs(test_dir+"star")
    os.makedirs(test_dir+"sky")

    create_fake(file_start=test_dir+"dither/dither",
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

    create_fake(file_start=test_dir+"star/star",
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

    create_fake(file_start=test_dir+"sky/sky",
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

    create_config(os.path.dirname(__file__)+"/PynPoint_config.ini")

def teardown_module():
    test_dir = os.path.dirname(__file__)+"/"

    for i in range(4):
        os.remove(test_dir+'dither/dither'+str(i+1).zfill(2)+'.fits')
        os.remove(test_dir+'star/star'+str(i+1).zfill(2)+'.fits')
        os.remove(test_dir+'sky/sky'+str(i+1).zfill(2)+'.fits')

    os.rmdir(test_dir+'dither')
    os.rmdir(test_dir+'star')
    os.rmdir(test_dir+'sky')

    os.remove(os.path.dirname(__file__)+"/PynPoint_database.hdf5")
    os.remove(os.path.dirname(__file__)+"/PynPoint_config.ini")

class TestBackgroundSubtraction(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def test_simple_background_subraction(self):

        print self.test_dir

        read = FitsReadingModule(name_in="read",
                                 image_tag="read",
                                 input_dir=self.test_dir+"dither")

        self.pipeline.add_module(read)

        simple = SimpleBackgroundSubtractionModule(shift=20,
                                                   name_in="simple",
                                                   image_in_tag="read",
                                                   image_out_tag="simple")

        self.pipeline.add_module(simple)

        self.pipeline.run()

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 74, 24], 0.05304008435511765, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010033896953157959, rtol=limit, atol=0.)

        data = self.pipeline.get_data("simple")
        assert np.allclose(data[0, 74, 74], -0.05288064325101517, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.7755575615628916e-22, rtol=limit, atol=0.)

    def test_mean_background_subraction(self):

        mean1 = MeanBackgroundSubtractionModule(shift=None,
                                                cubes=1,
                                                name_in="mean1",
                                                image_in_tag="read",
                                                image_out_tag="mean1")

        self.pipeline.add_module(mean1)

        mean2 = MeanBackgroundSubtractionModule(shift=20,
                                                cubes=1,
                                                name_in="mean2",
                                                image_in_tag="read",
                                                image_out_tag="mean2")

        self.pipeline.add_module(mean2)

        self.pipeline.run()

        data = self.pipeline.get_data("mean1")
        assert np.allclose(data[0, 74, 24], 0.0530465391626132, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.3970872216676808e-07, rtol=limit, atol=0.)

        data = self.pipeline.get_data("mean2")
        assert np.allclose(data[0, 74, 24], 0.0530465391626132, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.3970872216676808e-07, rtol=limit, atol=0.)

    def test_dithering_background(self):

        pca_dither1 = DitheringBackgroundModule(name_in="pca_dither1",
                                                image_in_tag="read",
                                                image_out_tag="pca_dither1",
                                                center=None,
                                                cubes=None,
                                                size=0.8,
                                                gaussian=0.1,
                                                subframe=0.1,
                                                pca_number=5,
                                                mask_star=0.1,
                                                crop=True,
                                                prepare=True,
                                                pca_background=True,
                                                combine="pca")

        self.pipeline.add_module(pca_dither1)

        pca_dither2 = DitheringBackgroundModule(name_in="pca_dither2",
                                                image_in_tag="read",
                                                image_out_tag="pca_dither2",
                                                center=((25., 75.), (75., 75.), (75., 25.), (25., 25.)),
                                                cubes=1,
                                                size=0.8,
                                                gaussian=0.1,
                                                pca_number=5,
                                                mask_star=0.1,
                                                crop=True,
                                                prepare=True,
                                                pca_background=True,
                                                combine="pca")

        self.pipeline.add_module(pca_dither2)

        self.pipeline.run()

        data = self.pipeline.get_data("pca_dither1")
        assert np.allclose(data[0, 13, 13], 0.05300595399147854, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.0012752305968659608, rtol=1e-6, atol=0.)

        data = self.pipeline.get_data("pca_dither2")
        assert np.allclose(data[0, 13, 13], 0.05300595400110353, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.001275230597238928, rtol=1e-3, atol=0.)

    def test_nodding_background(self):

        read1 = FitsReadingModule(name_in="read1",
                                  image_tag="star",
                                  input_dir=self.test_dir+"star")

        self.pipeline.add_module(read1)

        read2 = FitsReadingModule(name_in="read2",
                                  image_tag="sky",
                                  input_dir=self.test_dir+"sky")

        self.pipeline.add_module(read2)

        nodding = NoddingBackgroundModule(name_in="nodding",
                                          sky_in_tag="sky",
                                          science_in_tag="star",
                                          image_out_tag="nodding",
                                          mode="both")

        self.pipeline.add_module(nodding)

        self.pipeline.run()

        data = self.pipeline.get_data("star")
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)

        data = self.pipeline.get_data("sky")
        assert np.allclose(data[0, 50, 50], -7.613171257478652e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 8.937360237872607e-07, rtol=limit, atol=0.)

        data = self.pipeline.get_data("nodding")
        assert np.allclose(data[0, 50, 50], 0.09806026673451182, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.942251790089106e-05, rtol=limit, atol=0.)
