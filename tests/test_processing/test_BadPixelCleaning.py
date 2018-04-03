import os
import warnings

import h5py
import numpy as np

from PynPoint import Pypeline
from PynPoint.Core.DataIO import DataStorage
from PynPoint.ProcessingModules.BadPixelCleaning import BadPixelSigmaFilterModule, BadPixelMapModule, \
                                                        BadPixelInterpolationModule, BadPixelRefinementModule

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    file_in = os.path.dirname(__file__) + "/PynPoint_database.hdf5"
    config_file = os.path.dirname(__file__) + "/PynPoint_config.ini"

    np.random.seed(1)
    images = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))
    dark = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))
    flat = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))

    images[0, 10, 10] = 1.
    images[0, 12, 12] = 1.
    images[0, 14, 14] = 1.
    images[0, 20, 20] = 1.
    images[0, 22, 22] = 1.
    images[0, 24, 24] = 1.
    dark[:, 10, 10] = 1.
    dark[:, 12, 12] = 1.
    dark[:, 14, 14] = 1.
    flat[:, 20, 20] = -1.
    flat[:, 22, 22] = -1.
    flat[:, 24, 24] = -1.

    h5f = h5py.File(file_in, "w")
    h5f.create_dataset("images", data=images)
    h5f.create_dataset("dark", data=dark)
    h5f.create_dataset("flat", data=flat)
    h5f.create_dataset("header_images/STAR_POSITION", data=np.full((40, 2), 50.))
    h5f.close()

    f = open(config_file, 'w')
    f.write('[header]\n\n')
    f.write('INSTRUMENT: INSTRUME\n')
    f.write('NFRAMES: NAXIS3\n')
    f.write('EXP_NO: ESO DET EXP NO\n')
    f.write('NDIT: ESO DET NDIT\n')
    f.write('PARANG_START: ESO ADA POSANG\n')
    f.write('PARANG_END: ESO ADA POSANG END\n')
    f.write('DITHER_X: ESO SEQ CUMOFFSETX\n')
    f.write('DITHER_Y: None\n')
    f.write('DIT: None\n')
    f.write('LATITUDE: None\n')
    f.write('LONGITUDE: None\n')
    f.write('PUPIL: None\n')
    f.write('DATE: None\n')
    f.write('RA: None\n')
    f.write('DEC: None\n\n')
    f.write('[settings]\n\n')
    f.write('PIXSCALE: 0.01\n')
    f.write('MEMORY: 100\n')
    f.write('CPU: 1')
    f.close()

def teardown_module():
    file_in = os.path.dirname(__file__) + "/PynPoint_database.hdf5"
    config_file = os.path.dirname(__file__) + "/PynPoint_config.ini"

    os.remove(file_in)
    os.remove(config_file)

class TestBadPixelCleaning(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def test_bad_pixel_sigma_filter(self):

        sigma = BadPixelSigmaFilterModule(name_in="sigma",
                                          image_in_tag="images",
                                          image_out_tag="sigma",
                                          box=9,
                                          sigma=5,
                                          iterate=1)

        self.pipeline.add_module(sigma)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["sigma"]

        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 0.025022559679385093, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], 0.024962143884217046, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 6.721637736047109e-07, rtol=limit, atol=0.)

        storage.close_connection()

    def test_bad_pixel_map(self):

        bp_map = BadPixelMapModule(name_in="bp_map",
                                   dark_in_tag="dark",
                                   flat_in_tag="flat",
                                   bp_map_out_tag="bp_map",
                                   dark_threshold=0.99,
                                   flat_threshold=-0.99)

        self.pipeline.add_module(bp_map)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["bp_map"]

        assert data[0, 0] == 1.
        assert data[30, 30] == 1.
        assert data[10, 10] == 0.
        assert data[12, 12] == 0.
        assert data[14, 14] == 0.
        assert data[20, 20] == 0.
        assert data[22, 22] == 0.
        assert data[24, 24] == 0.
        assert np.mean(data) == 0.9993

        storage.close_connection()

    def test_bad_pixel_interpolation(self):

        interpolation = BadPixelInterpolationModule(name_in="interpolation",
                                                    image_in_tag="images",
                                                    bad_pixel_map_tag="bp_map",
                                                    image_out_tag="interpolation",
                                                    iterations=100)

        self.pipeline.add_module(interpolation)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["interpolation"]

        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 1.0139222106683477e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], -4.686852973820094e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 3.0499629451215465e-07, rtol=limit, atol=0.)

        storage.close_connection()

    def test_bad_pixel_refinement(self):

        refinement = BadPixelRefinementModule(name_in="refinement",
                                              image_in_tag="images",
                                              bad_pixel_map_tag="bp_map",
                                              image_out_tag="refinement",
                                              box_size=5,
                                              sigma=4.,
                                              iterations=100)

        self.pipeline.add_module(refinement)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["refinement"]

        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 3.4572557271785087e-06, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], -9.694818774930603e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 3.001502328056615e-07, rtol=limit, atol=0.)

        storage.close_connection()
