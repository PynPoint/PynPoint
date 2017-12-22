"""
Test cases for end-to-end data reduction.
"""

import os
import numpy as np

from PynPoint import Pypeline
from PynPoint.core.DataIO import DataStorage, InputPort
from PynPoint.io_modules import ReadFitsCubesDirectory
from PynPoint.processing_modules import RemoveLastFrameModule, PSFSubtractionModule, \
                                        AngleCalculationModule, CutTopLinesModule, \
                                        MeanBackgroundSubtractionModule, RemoveFramesModule, \
                                        BadPixelCleaningSigmaFilterModule, StarExtractionModule, \
                                        StarAlignmentModule, StackAndSubsetModule


class TestEndToEnd(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__)
        self.pipeline = Pypeline(self.test_dir, self.test_dir+"/test_data/adi/", self.test_dir)

    def test_read(self):
        read_fits = ReadFitsCubesDirectory(name_in="read_fits",
                                           image_tag="im",
                                           force_overwrite_in_databank=True)

        self.pipeline.add_module(read_fits)
        self.pipeline.run_module("read_fits")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im"]

        assert data[0, 0, 0] == -0.00018744786341144191
        assert np.mean(data) == 9.4480170917766312e-05
        assert data.shape == (84, 102, 100)

        storage.close_connection()

    def test_remove_last(self):
        remove_last = RemoveLastFrameModule(name_in="remove_last",
                                            image_in_tag="im",
                                            image_out_tag="im_last")

        self.pipeline.add_module(remove_last)
        self.pipeline.run_module("remove_last")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_last"]

        assert data[0, 0, 0] == -0.00018744786341144191
        assert np.mean(data) == 9.9204179463654638e-05
        assert data.shape == (80, 102, 100)

        storage.close_connection()

    def test_parang(self):
        angle = AngleCalculationModule(name_in="angle",
                                       data_tag="im_last")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        port = InputPort("im_last", storage)

        assert port.get_attribute("Used_Files")[0] == self.test_dir+'/test_data/adi/data01.fits'
        assert port.get_attribute("NEW_PARA")[1] == 1.1904761904761905

        port.close_port()
        storage.close_connection()

    def test_cut_lines(self):
        cut_lines = CutTopLinesModule(name_in="cut_lines",
                                      image_in_tag="im_last",
                                      image_out_tag="im_cut",
                                      num_lines=2)

        self.pipeline.add_module(cut_lines)
        self.pipeline.run_module("cut_lines")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_cut"]

        assert data[0, 0, 0] == -0.00018744786341144191
        assert np.mean(data) == 0.00010118826305292773
        assert data.shape == (80, 100, 100)

        storage.close_connection()

    def test_background(self):
        background = MeanBackgroundSubtractionModule(star_pos_shift=None,
                                                     cubes_per_position=1,
                                                     name_in="background",
                                                     image_in_tag="im_cut",
                                                     image_out_tag="im_bg")

        self.pipeline.add_module(background)
        self.pipeline.run_module("background")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_bg"]

        assert data[0, 0, 0] == -0.00022134946299066599
        assert np.mean(data) == 1.3911238380798353e-07
        assert data.shape == (80, 100, 100)

        storage.close_connection()

    def test_bad_pixel(self):
        bad_pixel = BadPixelCleaningSigmaFilterModule(name_in="bad_pixel",
                                                      image_in_tag="im_bg",
                                                      image_out_tag="im_bp",
                                                      box=9,
                                                      sigma=8,
                                                      iterate=3)

        self.pipeline.add_module(bad_pixel)
        self.pipeline.run_module("bad_pixel")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_bp"]

        assert data[0, 0, 0] == -0.00022134946299066599
        assert np.mean(data) == 1.3911238380798353e-07
        assert data.shape == (80, 100, 100)

        storage.close_connection()

    def test_star(self):
        star = StarExtractionModule(name_in="star",
                                    image_in_tag="im_bp",
                                    image_out_tag="im_star",
                                    psf_size=40,
                                    psf_size_as_pixel_resolution=True,
                                    fwhm_star=4)

        self.pipeline.add_module(star)
        self.pipeline.run_module("star")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_star"]

        assert data[0, 0, 0] == -7.4294323405466921e-05
        assert np.mean(data) == 0.00063168185861750764
        assert data.shape == (80, 40, 40)

        storage.close_connection()

    def test_center(self):
        center = StarAlignmentModule(name_in="center",
                                     image_in_tag="im_star",
                                     ref_image_in_tag=None,
                                     image_out_tag="im_center",
                                     interpolation="spline",
                                     accuracy=10,
                                     resize=5)

        self.pipeline.add_module(center)
        self.pipeline.run_module("center")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_center"]

        assert data[1, 0, 0] == -7.7284646607979651e-06
        assert data[16, 0, 0] == 6.5481930420497295e-06
        assert data[50, 0, 0] == -1.8830981620448768e-06
        assert data[67, 0, 0] == -5.7283610498600854e-06
        assert np.mean(data) == 2.5267274344700309e-05
        assert data.shape == (80, 200, 200)

        storage.close_connection()

    def test_remove_frames(self):
        remove_frames = RemoveFramesModule((0, 15, 49, 66),
                                           name_in="remove_frames",
                                           image_in_tag="im_center",
                                           image_out_tag="im_remove")

        self.pipeline.add_module(remove_frames)
        self.pipeline.run_module("remove_frames")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_remove"]

        assert data[0, 0, 0] == -7.7284646607979651e-06
        assert data[14, 0, 0] == 6.5481930420497295e-06
        assert data[47, 0, 0] == -1.8830981620448768e-06
        assert data[63, 0, 0] == -5.7283610498600854e-06
        assert np.mean(data) == 2.5268782670694567e-05
        assert data.shape == (76, 200, 200)

        storage.close_connection()

    def test_subset(self):
        subset = StackAndSubsetModule(name_in="subset",
                                      image_in_tag="im_remove",
                                      image_out_tag="im_subset",
                                      random_subset=38,
                                      stacking=2)

        self.pipeline.add_module(subset)
        self.pipeline.run_module("subset")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_subset"]

        assert data[0, 0, 0] == -6.1490341969019138e-06
        assert np.mean(data) == 2.5268782670694567e-05
        assert data.shape == (38, 200, 200)

        storage.close_connection()

    def test_pca(self):
        pca = PSFSubtractionModule(name_in="pca",
                                   pca_number=2,
                                   images_in_tag="im_subset",
                                   reference_in_tag="im_subset",
                                   res_arr_out_tag="res_arr",
                                   res_arr_rot_out_tag="res_rot",
                                   res_mean_tag="res_mean",
                                   res_median_tag="res_median",
                                   res_var_tag="res_var",
                                   res_rot_mean_clip_tag="res_rot_mean_clip",
                                   extra_rot=0.0,
                                   cent_size=0.1)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca")

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["res_mean"]

        assert data[154, 99] == 0.00040503841890726107
        assert np.mean(data) == -3.5359607872233667e-09
        assert data.shape == (200, 200)

        storage.close_connection()
