import os

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.psfpreparation import PSFpreparationModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule, ClassicalADIModule
from pynpoint.util.tests import create_config, create_fake_data, remove_test_data


class TestPsfSubtraction:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_fake_data(self.test_dir+'science')
        create_fake_data(self.test_dir+'reference')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['science', 'reference'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read1',
                                   image_tag='science',
                                   input_dir=self.test_dir+'science')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        data = self.pipeline.get_data('science')
        assert np.sum(data) == pytest.approx(11.012854046962481, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        self.pipeline.set_attribute('science', 'PARANG', np.linspace(0., 180., 10), static=False)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='reference',
                                   input_dir=self.test_dir+'reference')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('reference')
        assert np.sum(data) == pytest.approx(11.012854046962481, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

    def test_psf_preparation(self) -> None:

        module = PSFpreparationModule(name_in='prep1',
                                      image_in_tag='science',
                                      image_out_tag='science_prep',
                                      mask_out_tag=None,
                                      norm=False,
                                      resize=None,
                                      cent_size=0.05,
                                      edge_size=1.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('prep1')

        data = self.pipeline.get_data('science_prep')
        assert np.sum(data) == pytest.approx(5.029285028467547, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        module = PSFpreparationModule(name_in='prep2',
                                      image_in_tag='reference',
                                      image_out_tag='reference_prep',
                                      mask_out_tag=None,
                                      norm=False,
                                      resize=None,
                                      cent_size=0.05,
                                      edge_size=1.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('prep2')

        data = self.pipeline.get_data('reference_prep')
        assert np.sum(data) == pytest.approx(5.029285028467547, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

    def test_classical_adi(self) -> None:

        module = ClassicalADIModule(threshold=None,
                                    nreference=None,
                                    residuals='mean',
                                    extra_rot=0.,
                                    name_in='cadi1',
                                    image_in_tag='science_prep',
                                    res_out_tag='cadi_res',
                                    stack_out_tag='cadi_stack')

        self.pipeline.add_module(module)
        self.pipeline.run_module('cadi1')

        data = self.pipeline.get_data('cadi_res')
        assert np.sum(data) == pytest.approx(0.8381625719865213, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        data = self.pipeline.get_data('cadi_stack')
        assert np.sum(data) == pytest.approx(0.08395606034388256, rel=self.limit, abs=0.)
        assert data.shape == (1, 21, 21)

    def test_classical_adi_threshold(self) -> None:

        module = ClassicalADIModule(threshold=(0.1, 0.03, 1.),
                                    nreference=5,
                                    residuals='median',
                                    extra_rot=0.,
                                    name_in='cadi2',
                                    image_in_tag='science_prep',
                                    res_out_tag='cadi_res',
                                    stack_out_tag='cadi_stack')

        self.pipeline.add_module(module)
        self.pipeline.run_module('cadi2')

        data = self.pipeline.get_data('cadi_res')
        assert np.sum(data) == pytest.approx(0.7158207863548083, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        data = self.pipeline.get_data('cadi_stack')
        assert np.sum(data) == pytest.approx(0.07448334552227256, rel=self.limit, abs=0.)
        assert data.shape == (1, 21, 21)

    def test_psf_subtraction_pca_single(self) -> None:

        module = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                         name_in='pca_single',
                                         images_in_tag='science',
                                         reference_in_tag='science',
                                         res_mean_tag='res_mean_single',
                                         res_median_tag='res_median_single',
                                         res_weighted_tag='res_weighted_single',
                                         res_rot_mean_clip_tag='res_clip_single',
                                         res_arr_out_tag='res_arr_single',
                                         basis_out_tag='basis_single',
                                         extra_rot=45.,
                                         subtract_mean=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_single')

        data = self.pipeline.get_data('res_mean_single')
        assert np.sum(data) == pytest.approx(-0.00011857022709778602, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('res_median_single')
        assert np.sum(data) == pytest.approx(-0.002184868916566093, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('res_weighted_single')
        assert np.sum(data) == pytest.approx(0.08102176735226937, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        # data = self.pipeline.get_data('res_clip_single')
        # assert np.sum(data) == pytest.approx(7.09495495339349e-05, rel=self.limit, abs=0.)
        # assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('res_arr_single1')
        assert np.sum(data) == pytest.approx(-0.0002751385418691618, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        data = self.pipeline.get_data('basis_single')
        assert np.sum(data) == pytest.approx(0.09438697731322143, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

    def test_psf_subtraction_no_mean(self) -> None:

        module = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                         name_in='pca_no_mean',
                                         images_in_tag='science',
                                         reference_in_tag='science',
                                         res_mean_tag='res_mean_no_mean',
                                         res_median_tag=None,
                                         res_weighted_tag=None,
                                         res_rot_mean_clip_tag=None,
                                         res_arr_out_tag=None,
                                         basis_out_tag='basis_no_mean',
                                         extra_rot=0.,
                                         subtract_mean=False)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_no_mean')

        data = self.pipeline.get_data('res_mean_no_mean')
        assert np.sum(data) == pytest.approx(0.0005733657032555452, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('basis_no_mean')
        assert np.sum(data) == pytest.approx(5.118005177367776, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

    def test_psf_subtraction_ref(self) -> None:

        module = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                         name_in='pca_ref',
                                         images_in_tag='science',
                                         reference_in_tag='reference',
                                         res_mean_tag='res_mean_ref',
                                         res_median_tag=None,
                                         res_weighted_tag=None,
                                         res_rot_mean_clip_tag=None,
                                         res_arr_out_tag=None,
                                         basis_out_tag='basis_ref',
                                         extra_rot=0.,
                                         subtract_mean=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_ref')

        data = self.pipeline.get_data('res_mean_ref')
        assert np.sum(data) == pytest.approx(0.0005868283126528002, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('basis_ref')
        assert np.sum(data) == pytest.approx(0.0943869773132221, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

    def test_psf_subtraction_ref_no_mean(self) -> None:

        module = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                         name_in='pca_ref_no_mean',
                                         images_in_tag='science',
                                         reference_in_tag='reference',
                                         res_mean_tag='res_mean_ref_no_mean',
                                         res_median_tag=None,
                                         res_weighted_tag=None,
                                         res_rot_mean_clip_tag=None,
                                         res_arr_out_tag=None,
                                         basis_out_tag='basis_ref_no_mean',
                                         extra_rot=0.,
                                         subtract_mean=False)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_ref_no_mean')

        data = self.pipeline.get_data('res_mean_ref_no_mean')
        assert np.sum(data) == pytest.approx(0.0005733657032555494, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('basis_ref_no_mean')
        assert np.sum(data) == pytest.approx(5.118005177367774, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

    def test_psf_subtraction_pca_single_mask(self) -> None:

        pca = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                      name_in='pca_single_mask',
                                      images_in_tag='science_prep',
                                      reference_in_tag='science_prep',
                                      res_mean_tag='res_mean_single_mask',
                                      res_median_tag='res_median_single_mask',
                                      res_weighted_tag='res_weighted_single_mask',
                                      res_rot_mean_clip_tag='res_clip_single_mask',
                                      res_arr_out_tag='res_arr_single_mask',
                                      basis_out_tag='basis_single_mask',
                                      extra_rot=45.,
                                      subtract_mean=True)

        self.pipeline.add_module(pca)
        self.pipeline.run_module('pca_single_mask')

        data = self.pipeline.get_data('res_mean_single_mask')
        assert np.sum(data) == pytest.approx(0.00010696166038626307, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('res_median_single_mask')
        assert np.sum(data) == pytest.approx(-0.0021005307611346156, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('res_weighted_single_mask')
        assert np.sum(data) == pytest.approx(0.06014309988789256, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        # data = self.pipeline.get_data('res_clip_single_mask')
        # assert np.sum(data) == pytest.approx(9.35120662148806e-05, rel=self.limit, abs=0.)
        # assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('res_arr_single_mask1')
        assert np.sum(data) == pytest.approx(0.0006170872862547557, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        data = self.pipeline.get_data('basis_single_mask')
        assert np.sum(data) == pytest.approx(0.08411251293842359, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

    def test_psf_subtraction_no_mean_mask(self) -> None:

        module = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                         name_in='pca_no_mean_mask',
                                         images_in_tag='science_prep',
                                         reference_in_tag='science_prep',
                                         res_mean_tag='res_mean_no_mean_mask',
                                         res_median_tag=None,
                                         res_weighted_tag=None,
                                         res_rot_mean_clip_tag=None,
                                         res_arr_out_tag=None,
                                         basis_out_tag='basis_no_mean_mask',
                                         extra_rot=0.,
                                         subtract_mean=False)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_no_mean_mask')

        data = self.pipeline.get_data('res_mean_no_mean_mask')
        assert np.sum(data) == pytest.approx(2.3542359949502915e-05, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('basis_no_mean_mask')
        assert np.sum(data) == pytest.approx(5.655460951633232, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

    def test_psf_subtraction_ref_mask(self) -> None:

        module = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                         name_in='pca_ref_mask',
                                         images_in_tag='science_prep',
                                         reference_in_tag='reference_prep',
                                         res_mean_tag='res_mean_ref_mask',
                                         res_median_tag=None,
                                         res_weighted_tag=None,
                                         res_rot_mean_clip_tag=None,
                                         res_arr_out_tag=None,
                                         basis_out_tag='basis_ref_mask',
                                         extra_rot=0.,
                                         subtract_mean=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_ref_mask')

        data = self.pipeline.get_data('res_mean_ref_mask')
        assert np.sum(data) == pytest.approx(9.400558926815758e-06, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('basis_ref_mask')
        assert np.sum(data) == pytest.approx(0.08411251293842326, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

    def test_psf_subtraction_ref_no_mean_mask(self) -> None:

        module = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                         name_in='pca_ref_no_mean_mask',
                                         images_in_tag='science_prep',
                                         reference_in_tag='reference_prep',
                                         res_mean_tag='res_mean_ref_no_mean_mask',
                                         res_median_tag=None,
                                         res_weighted_tag=None,
                                         res_rot_mean_clip_tag=None,
                                         res_arr_out_tag=None,
                                         basis_out_tag='basis_ref_no_mean_mask',
                                         extra_rot=0.,
                                         subtract_mean=False)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_ref_no_mean_mask')

        data = self.pipeline.get_data('res_mean_ref_no_mean_mask')
        assert np.sum(data) == pytest.approx(2.354235994950671e-05, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

        data = self.pipeline.get_data('basis_ref_no_mean_mask')
        assert np.sum(data) == pytest.approx(5.655460951633233, rel=self.limit, abs=0.)
        assert data.shape == (2, 21, 21)

    def test_psf_subtraction_pca_multi(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        module = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                         name_in='pca_multi',
                                         images_in_tag='science',
                                         reference_in_tag='science',
                                         res_mean_tag='res_mean_multi',
                                         res_median_tag='res_median_multi',
                                         res_weighted_tag='res_weighted_multi',
                                         res_rot_mean_clip_tag='res_clip_multi',
                                         res_arr_out_tag=None,
                                         basis_out_tag='basis_multi',
                                         extra_rot=45.,
                                         subtract_mean=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_multi')

        data_single = self.pipeline.get_data('res_mean_single')
        data_multi = self.pipeline.get_data('res_mean_multi')
        assert data_single.shape == data_multi.shape

        assert data_single[data_single > 1e-12] == \
            pytest.approx(data_multi[data_multi > 1e-12], rel=self.limit, abs=0.)

        data_single = self.pipeline.get_data('res_median_single')
        data_multi = self.pipeline.get_data('res_median_multi')
        assert data_single.shape == data_multi.shape

        assert data_single[data_single > 1e-12] == \
            pytest.approx(data_multi[data_multi > 1e-12], rel=self.limit, abs=0.)

        data_single = self.pipeline.get_data('res_weighted_single')
        data_multi = self.pipeline.get_data('res_weighted_multi')
        assert data_single.shape == data_multi.shape

        assert data_single[data_single > 1e-12] == \
            pytest.approx(data_multi[data_multi > 1e-12], rel=self.limit, abs=0.)

        data_single = self.pipeline.get_data('basis_single')
        data_multi = self.pipeline.get_data('basis_multi')
        assert data_single.shape == data_multi.shape

        assert data_single[data_single > 1e-12] == \
            pytest.approx(data_multi[data_multi > 1e-12], rel=self.limit, abs=0.)

    def test_psf_subtraction_pca_multi_mask(self) -> None:

        database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
        database['config'].attrs['CPU'] = 4

        module = PcaPsfSubtractionModule(pca_numbers=range(1, 3),
                                         name_in='pca_multi_mask',
                                         images_in_tag='science_prep',
                                         reference_in_tag='science_prep',
                                         res_mean_tag='res_mean_multi_mask',
                                         res_median_tag='res_median_multi_mask',
                                         res_weighted_tag='res_weighted_multi_mask',
                                         res_rot_mean_clip_tag='res_clip_multi_mask',
                                         res_arr_out_tag=None,
                                         basis_out_tag='basis_multi_mask',
                                         extra_rot=45.,
                                         subtract_mean=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_multi_mask')

        data_single = self.pipeline.get_data('res_mean_single_mask')
        data_multi = self.pipeline.get_data('res_mean_multi_mask')
        assert data_single.shape == data_multi.shape

        assert data_single[data_single > 1e-12] == \
            pytest.approx(data_multi[data_multi > 1e-12], rel=self.limit, abs=0.)

        data_single = self.pipeline.get_data('res_median_single_mask')
        data_multi = self.pipeline.get_data('res_median_multi_mask')
        assert data_single.shape == data_multi.shape

        assert data_single[data_single > 1e-12] == \
            pytest.approx(data_multi[data_multi > 1e-12], rel=self.limit, abs=0.)

        data_single = self.pipeline.get_data('res_weighted_single_mask')
        data_multi = self.pipeline.get_data('res_weighted_multi_mask')
        assert data_single.shape == data_multi.shape

        assert data_single[data_single > 1e-12] == \
            pytest.approx(data_multi[data_multi > 1e-12], rel=self.limit, abs=0.)

        data_single = self.pipeline.get_data('basis_single_mask')
        data_multi = self.pipeline.get_data('basis_multi_mask')
        assert data_single.shape == data_multi.shape
        assert data_single == pytest.approx(data_multi, rel=self.limit, abs=0.)

    def test_psf_subtraction_len_parang(self) -> None:

        database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
        database['config'].attrs['CPU'] = 1

        parang = self.pipeline.get_data('header_science/PARANG')
        self.pipeline.set_attribute('science_prep', 'PARANG', np.append(parang, 0.), static=False)

        module = PcaPsfSubtractionModule(pca_numbers=[1, ],
                                         name_in='pca_len_parang',
                                         images_in_tag='science_prep',
                                         reference_in_tag='science_prep',
                                         res_mean_tag='res_mean_len_parang',
                                         extra_rot=0.)

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('pca_len_parang')

        assert str(error.value) == 'The number of images (10) is not equal to the number of ' \
                                   'parallactic angles (11).'
