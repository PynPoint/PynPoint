"""
Test cases for examples in the documentation.
"""

import os

import numpy as np

from PynPoint import Pypeline
from PynPoint.core.DataIO import DataStorage
from PynPoint.io_modules import FitsReadingModule, FitsWritingModule
from PynPoint.processing_modules import BadPixelCleaningSigmaFilterModule, \
     DarkSubtractionModule, FlatSubtractionModule, CutTopLinesModule, \
     AngleCalculationModule, MeanBackgroundSubtractionModule, \
     StarExtractionModule, StarAlignmentModule, PSFSubtractionModule, \
     StackAndSubsetModule, RemoveLastFrameModule


class TestDocumentation(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__)
        self.pipeline = Pypeline(self.test_dir, self.test_dir+"/test_data/adi/", self.test_dir)

    def test_docs(self):

        reading_data = FitsReadingModule(name_in="Fits_reading",
                                         image_tag="im_arr")

        self.pipeline.add_module(reading_data)

        reading_dark = FitsReadingModule(name_in="Dark_reading",
                                         input_dir=self.test_dir+"/test_data/dark",
                                         image_tag="dark_arr")

        self.pipeline.add_module(reading_dark)

        reading_flat = FitsReadingModule(name_in="Flat_reading",
                                         input_dir=self.test_dir+"/test_data/flat",
                                         image_tag="flat_arr")

        self.pipeline.add_module(reading_flat)

        remove_last = RemoveLastFrameModule(name_in="last_frame",
                                            image_in_tag="im_arr",
                                            image_out_tag="im_arr_last")

        self.pipeline.add_module(remove_last)

        cutting = CutTopLinesModule(name_in="NACO_cutting",
                                    image_in_tag="im_arr_last",
                                    image_out_tag="im_arr_cut",
                                    num_lines=2)

        self.pipeline.add_module(cutting)

        dark_sub = DarkSubtractionModule(name_in="dark_subtraction",
                                         image_in_tag="im_arr_cut",
                                         dark_in_tag="dark_arr",
                                         image_out_tag="dark_sub_arr")

        flat_sub = FlatSubtractionModule(name_in="flat_subtraction",
                                         image_in_tag="dark_sub_arr",
                                         flat_in_tag="flat_arr",
                                         image_out_tag="flat_sub_arr")

        self.pipeline.add_module(dark_sub)
        self.pipeline.add_module(flat_sub)

        bg_subtraction = MeanBackgroundSubtractionModule(star_pos_shift=None,
                                                         cubes_per_position=1,
                                                         name_in="background_subtraction",
                                                         image_in_tag="flat_sub_arr",
                                                         image_out_tag="bg_cleaned_arr")


        self.pipeline.add_module(bg_subtraction)

        bp_cleaning = BadPixelCleaningSigmaFilterModule(name_in="sigma_filtering",
                                                        image_in_tag="bg_cleaned_arr",
                                                        image_out_tag="bp_cleaned_arr")

        self.pipeline.add_module(bp_cleaning)

        extraction = StarExtractionModule(name_in="star_cutting",
                                          image_in_tag="bp_cleaned_arr",
                                          image_out_tag="im_arr_extract",
                                          image_size=1.,
                                          fwhm_star=0.1)

        # Required for ref_image_in_tag in StarAlignmentModule, otherwise a random frame is used
        ref_extract = StarExtractionModule(name_in="star_cut_ref",
                                           image_in_tag="bp_cleaned_arr",
                                           image_out_tag="im_arr_ref",
                                           image_size=1.,
                                           fwhm_star=0.1)

        alignment = StarAlignmentModule(name_in="star_align",
                                        image_in_tag="im_arr_extract",
                                        ref_image_in_tag="im_arr_ref",
                                        image_out_tag="im_arr_aligned",
                                        accuracy=100,
                                        resize=2)

        self.pipeline.add_module(extraction)
        self.pipeline.add_module(ref_extract)
        self.pipeline.add_module(alignment)

        angle_calc = AngleCalculationModule(name_in="angle_calculation",
                                            data_tag="im_arr_aligned")

        self.pipeline.add_module(angle_calc)

        subset = StackAndSubsetModule(name_in="stacking_subset",
                                      image_in_tag="im_arr_aligned",
                                      image_out_tag="im_arr_stacked",
                                      random_subset=None,
                                      stacking=4)

        self.pipeline.add_module(subset)

        psf_sub = PSFSubtractionModule(pca_number=5,
                                       name_in="PSF_subtraction",
                                       images_in_tag="im_arr_stacked",
                                       reference_in_tag="im_arr_stacked",
                                       res_mean_tag="res_mean")

        self.pipeline.add_module(psf_sub)

        writing = FitsWritingModule(name_in="Fits_writing",
                                    file_name="test.fits",
                                    data_tag="res_mean")

        self.pipeline.add_module(writing)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["im_arr"]
        assert data[0, 61, 39] == -0.00022889163546536875

        data = storage.m_data_bank["dark_arr"]
        assert data[0, 61, 39] == 2.368170995592123e-05

        data = storage.m_data_bank["flat_arr"]
        assert data[0, 61, 39] == 0.98703416941301647

        data = storage.m_data_bank["im_arr_last"]
        assert data[0, 61, 39] == -0.00022889163546536875

        data = storage.m_data_bank["im_arr_cut"]
        assert data[0, 61, 39] == -0.00022889163546536875

        data = storage.m_data_bank["dark_sub_arr"]
        assert data[0, 61, 39] == -0.00021601281733413911

        data = storage.m_data_bank["flat_sub_arr"]
        assert data[0, 61, 39] == -0.00021553814660050282

        data = storage.m_data_bank["bg_cleaned_arr"]
        assert data[0, 61, 39] == -0.00013038694003957227

        data = storage.m_data_bank["bp_cleaned_arr"]
        assert data[0, 61, 39] == -0.00013038694003957227

        data = storage.m_data_bank["im_arr_extract"]
        assert data[0, 31, 20] == -3.9392607130869333e-05

        data = storage.m_data_bank["im_arr_aligned"]
        assert data[0, 61, 39] == 0.00021600121168847015

        data = storage.m_data_bank["im_arr_stacked"]
        assert data[0, 61, 39] == 8.2429659114370023e-05

        data = storage.m_data_bank["res_mean"]
        assert data[61, 39] == -4.4710338563281608e-05
        assert np.mean(data) == 5.196570975891392e-08
        assert data.shape == (72, 72)

        storage.close_connection()
