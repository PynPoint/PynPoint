import os
import warnings

import h5py
import numpy as np

from PynPoint import Pypeline
from PynPoint.Core.DataIO import DataStorage
from PynPoint.ProcessingModules.BackgroundSubtraction import MeanBackgroundSubtractionModule, SimpleBackgroundSubtractionModule, \
                                                             PCABackgroundPreparationModule, PCABackgroundSubtractionModule, \
                                                             DitheringBackgroundModule, NoddingBackgroundModule

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    file_in = os.path.dirname(__file__) + "/PynPoint_database.hdf5"
    config_file = os.path.dirname(__file__) + "/PynPoint_config.ini"

    np.random.seed(1)
    images = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))
    sky = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))

    h5f = h5py.File(file_in, "w")
    dset = h5f.create_dataset("images", data=images)
    dset.attrs['PIXSCALE'] = 0.01
    h5f.create_dataset("header_images/NFRAMES", data=[10, 10, 10, 10])
    h5f.create_dataset("header_images/EXP_NO", data=[1, 3, 5, 7])
    h5f.create_dataset("header_images/DITHER_X", data=[5, 5, -5, -5])
    h5f.create_dataset("header_images/DITHER_Y", data=[5, -5, -5, 5])
    h5f.create_dataset("header_images/STAR_POSITION", data=np.full((10, 2), 40.))
    dset = h5f.create_dataset("sky", data=sky)
    dset.attrs['PIXSCALE'] = 0.01
    h5f.create_dataset("header_sky/NFRAMES", data=[10, 10, 10, 10])
    h5f.create_dataset("header_sky/EXP_NO", data=[2, 4, 6, 8])
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
    f.write('DITHER_Y: ESO SEQ CUMOFFSETY\n\n')
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

class TestBackgroundSubtraction(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def test_simple_background_subraction(self):

        simple = SimpleBackgroundSubtractionModule(shift=1,
                                                   name_in="simple",
                                                   image_in_tag="images",
                                                   image_out_tag="simple")

        self.pipeline.add_module(simple)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["simple"]
        assert np.allclose(data[0, 10, 10], 0.00016307583939841944, rtol=limit)
        assert np.allclose(np.mean(data), 1.7347234759768072e-23, rtol=limit)

        storage.close_connection()

    def test_mean_background_subraction(self):

        mean1 = MeanBackgroundSubtractionModule(shift=None,
                                                cubes=1,
                                                name_in="mean1",
                                                image_in_tag="images",
                                                image_out_tag="mean1")

        self.pipeline.add_module(mean1)

        mean2 = MeanBackgroundSubtractionModule(shift=10,
                                                cubes=1,
                                                name_in="mean2",
                                                image_in_tag="images",
                                                image_out_tag="mean2")

        self.pipeline.add_module(mean2)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["mean1"]
        assert np.allclose(data[0, 10, 10], 0.0001881700200690493, rtol=limit)
        assert np.allclose(np.mean(data), 6.468177714836324e-08, rtol=limit)

        data = storage.m_data_bank["mean2"]
        assert np.allclose(data[0, 10, 10], 0.0001881700200690493, rtol=limit)
        assert np.allclose(np.mean(data), 6.468177714836324e-08, rtol=limit)

        storage.close_connection()

    def test_pca_background(self):

        pca_prep = PCABackgroundPreparationModule(dither=(4, 1, 0),
                                                  name_in="pca_prep",
                                                  image_in_tag="images",
                                                  star_out_tag="star",
                                                  background_out_tag="background")

        self.pipeline.add_module(pca_prep)

        pca_bg = PCABackgroundSubtractionModule(pca_number=5,
                                                mask=0.3,
                                                name_in="pca_bg",
                                                star_in_tag="star",
                                                background_in_tag="background",
                                                subtracted_out_tag="subtracted",
                                                residuals_out_tag="residuals")

        self.pipeline.add_module(pca_bg)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["star"]
        assert np.allclose(data[0, 10, 10], 0.0001881700200690493, rtol=limit)
        assert np.allclose(np.mean(data), 3.137393482985464e-07, rtol=limit)

        data = storage.m_data_bank["background"]
        assert np.allclose(data[0, 10, 10], 9.38719992395586e-05, rtol=limit)
        assert np.allclose(np.mean(data), 5.782411586589357e-23, rtol=limit)

        data = storage.m_data_bank["subtracted"]
        assert np.allclose(data[0, 10, 10], 0.00017730626029598609, rtol=limit)
        assert np.allclose(np.mean(data), 3.102053539454623e-07, rtol=limit)

        data = storage.m_data_bank["residuals"]
        assert np.allclose(data[0, 10, 10], 1.0863759773063205e-05, rtol=limit)
        assert np.allclose(np.mean(data), 1.3019973427207451e-08, rtol=limit)

        storage.close_connection()

    def test_dithering_background(self):

        pca_dither1 = DitheringBackgroundModule(name_in="pca_dither1",
                                                image_in_tag="images",
                                                image_out_tag="pca_dither1",
                                                center=None,
                                                cubes=None,
                                                size=0.8,
                                                gaussian=0.15,
                                                pca_number=5,
                                                mask=0.3,
                                                crop=True,
                                                prepare=True,
                                                pca_background=True,
                                                combine="pca")

        self.pipeline.add_module(pca_dither1)

        pca_dither2 = DitheringBackgroundModule(name_in="pca_dither2",
                                                image_in_tag="images",
                                                image_out_tag="pca_dither2",
                                                center=((55., 55.), (55., 45.), (45., 45.), (45., 55.)),
                                                cubes=1,
                                                size=0.8,
                                                gaussian=0.15,
                                                pca_number=5,
                                                mask=0.3,
                                                crop=True,
                                                prepare=True,
                                                pca_background=True,
                                                combine="pca")

        self.pipeline.add_module(pca_dither2)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["pca_dither1"]
        assert np.allclose(data[0, 10, 10], 4.9091895841309806e-05, rtol=limit)
        assert np.allclose(np.mean(data), 4.381313106053087e-08, rtol=limit)

        data = storage.m_data_bank["pca_dither2"]
        assert np.allclose(data[0, 10, 10], 4.909189533975212e-05, rtol=limit)
        assert np.allclose(np.mean(data), 4.381304393943703e-08, rtol=limit)

        storage.close_connection()

    def test_nodding_background(self):

        nodding = NoddingBackgroundModule(name_in="nodding",
                                          sky_in_tag="sky",
                                          science_in_tag="images",
                                          image_out_tag="nodding",
                                          mode="both")

        self.pipeline.add_module(nodding)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["images"]
        assert np.allclose(data[0, 10, 10], 0.00012958496246258364, rtol=limit)
        assert np.allclose(np.mean(data), 2.9494781737579395e-07, rtol=limit)

        data = storage.m_data_bank["sky"]
        assert np.allclose(data[0, 10, 10], -3.7305891376589886e-05, rtol=limit)
        assert np.allclose(np.mean(data), 3.829912457736603e-07, rtol=limit)

        data = storage.m_data_bank["nodding"]
        assert np.allclose(data[0, 10, 10], 0.00016689085383917351, rtol=limit)
        assert np.allclose(np.mean(data), 9.439443523107406e-07, rtol=limit)

        storage.close_connection()
