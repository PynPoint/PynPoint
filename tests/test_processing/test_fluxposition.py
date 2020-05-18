import os

import h5py
import pytest
import numpy as np

from astropy.io import fits

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.fluxposition import FakePlanetModule, AperturePhotometryModule, \
                                             FalsePositiveModule, SimplexMinimizationModule, \
                                             MCMCsamplingModule, SystematicErrorModule
from pynpoint.processing.stacksubset import DerotateAndStackModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule
from pynpoint.util.tests import create_config, create_star_data, create_fake_data, remove_test_data


class TestFluxPosition:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_fake_data(self.test_dir+'adi')
        create_star_data(self.test_dir+'psf', npix=21, pos_star=10.)
        create_star_data(self.test_dir+'ref', npix=21, pos_star=10.)
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['adi', 'psf', 'ref'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read1',
                                   image_tag='adi',
                                   input_dir=self.test_dir+'adi')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        data = self.pipeline.get_data('adi')
        assert np.sum(data) == pytest.approx(10.112910611363908, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        self.pipeline.set_attribute('adi', 'PARANG', np.linspace(0., 180., 10), static=False)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='psf',
                                   input_dir=self.test_dir+'psf')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('psf')
        assert np.sum(data) == pytest.approx(10.012916896297398, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        module = FitsReadingModule(name_in='read3',
                                   image_tag='ref',
                                   input_dir=self.test_dir+'psf')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read3')

        data = self.pipeline.get_data('ref')
        assert np.sum(data) == pytest.approx(10.012916896297398, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

    def test_aperture_photometry(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 1

        module = AperturePhotometryModule(name_in='photometry1',
                                          image_in_tag='psf',
                                          phot_out_tag='photometry1',
                                          radius=0.1,
                                          position=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('photometry1')

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        module = AperturePhotometryModule(name_in='photometry2',
                                          image_in_tag='psf',
                                          phot_out_tag='photometry2',
                                          radius=0.1,
                                          position=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('photometry2')

        data = self.pipeline.get_data('photometry1')
        assert np.sum(data) == pytest.approx(9.8351400803546, rel=self.limit, abs=0.)
        assert data.shape == (10, 1)

        data_multi = self.pipeline.get_data('photometry2')
        assert data.shape == data_multi.shape
        # TODO Does not pass on Travis CI
        assert data == pytest.approx(data_multi, rel=1e-6, abs=0.)

    def test_aperture_photometry_position(self) -> None:

        module = AperturePhotometryModule(name_in='photometry3',
                                          image_in_tag='psf',
                                          phot_out_tag='photometry3',
                                          radius=0.1,
                                          position=(10., 10.))

        self.pipeline.add_module(module)
        self.pipeline.run_module('photometry3')

        data = self.pipeline.get_data('photometry3')
        assert np.sum(data) == pytest.approx(9.8351400803546, rel=self.limit, abs=0.)
        assert data.shape == (10, 1)

    def test_fake_planet(self) -> None:

        module = FakePlanetModule(position=(0.2, 180.),
                                  magnitude=2.5,
                                  psf_scaling=1.,
                                  interpolation='spline',
                                  name_in='fake',
                                  image_in_tag='adi',
                                  psf_in_tag='psf',
                                  image_out_tag='fake')

        self.pipeline.add_module(module)
        self.pipeline.run_module('fake')

        data = self.pipeline.get_data('fake')
        assert np.sum(data) == pytest.approx(11.113528472622743, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

    def test_psf_subtraction(self) -> None:

        module = PcaPsfSubtractionModule(pca_numbers=[1, ],
                                         name_in='pca',
                                         images_in_tag='fake',
                                         reference_in_tag='fake',
                                         res_mean_tag='res_mean',
                                         extra_rot=0.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca')

        data = self.pipeline.get_data('res_mean')
        assert np.sum(data) == pytest.approx(0.00028116589089421246, rel=self.limit, abs=0.)
        assert data.shape == (1, 21, 21)

    def test_false_positive(self) -> None:

        module = FalsePositiveModule(position=(10., 2.),
                                     aperture=0.06,
                                     ignore=True,
                                     name_in='false1',
                                     image_in_tag='res_mean',
                                     snr_out_tag='snr_fpf1',
                                     optimize=False)

        self.pipeline.add_module(module)
        self.pipeline.run_module('false1')

        data = self.pipeline.get_data('snr_fpf1')
        assert data[0, 1] == pytest.approx(2., rel=self.limit, abs=0.)
        assert data[0, 2] == pytest.approx(0.216, rel=self.limit, abs=0.)
        assert data[0, 3] == pytest.approx(180., rel=self.limit, abs=0.)
        assert data[0, 4] == pytest.approx(17.28619108898002, rel=self.limit, abs=0.)
        assert data[0, 5] == pytest.approx(2.6641329371344664e-07, rel=self.limit, abs=0.)
        assert data.shape == (1, 6)

    def test_false_positive_optimize(self) -> None:

        module = FalsePositiveModule(position=(10., 2.),
                                     aperture=0.06,
                                     ignore=True,
                                     name_in='false2',
                                     image_in_tag='res_mean',
                                     snr_out_tag='snr_fpf2',
                                     optimize=True,
                                     offset=0.1,
                                     tolerance=0.01)

        self.pipeline.add_module(module)
        self.pipeline.run_module('false2')

        data = self.pipeline.get_data('snr_fpf2')
        assert data[0, 1] == pytest.approx(1.9000985264778163, rel=self.limit, abs=0.)
        assert data[0, 2] == pytest.approx(0.21870706386646876, rel=self.limit, abs=0.)
        assert data[0, 3] == pytest.approx(179.45970353460103, rel=self.limit, abs=0.)
        assert data[0, 4] == pytest.approx(17.46063754556111, rel=self.limit, abs=0.)
        assert data[0, 5] == pytest.approx(2.4868277093896833e-07, rel=self.limit, abs=0.)
        assert data.shape == (1, 6)

    def test_simplex_minimization_hessian(self) -> None:

        module = SimplexMinimizationModule(name_in='simplex1',
                                           image_in_tag='fake',
                                           psf_in_tag='psf',
                                           res_out_tag='simplex_res',
                                           flux_position_tag='flux_position',
                                           position=(10, 3),
                                           magnitude=2.5,
                                           psf_scaling=-1.,
                                           merit='hessian',
                                           aperture=0.06,
                                           sigma=0.,
                                           tolerance=0.1,
                                           pca_number=1,
                                           cent_size=0.06,
                                           edge_size=None,
                                           extra_rot=0.,
                                           reference_in_tag=None,
                                           residuals='median',
                                           offset=3.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('simplex1')

        data = self.pipeline.get_data('simplex_res')
        assert np.sum(data) == pytest.approx(-0.014648717637221701, rel=self.limit, abs=0.)
        assert data.shape == (26, 21, 21)

        data = self.pipeline.get_data('flux_position')
        assert data[25, 0] == pytest.approx(9.985098975003805, rel=self.limit, abs=0.)
        assert data[25, 1] == pytest.approx(2.631290723593964, rel=self.limit, abs=0.)
        assert data[25, 2] == pytest.approx(0.19895555725663513, rel=self.limit, abs=0.)
        assert data[25, 3] == pytest.approx(179.88413646855793, rel=self.limit, abs=0.)
        assert data[25, 4] == pytest.approx(2.533497537532387, rel=self.limit, abs=0.)
        assert data.shape == (26, 6)

    def test_simplex_minimization_reference(self) -> None:

        module = SimplexMinimizationModule(name_in='simplex2',
                                           image_in_tag='fake',
                                           psf_in_tag='psf',
                                           res_out_tag='simplex_res_ref',
                                           flux_position_tag='flux_position_ref',
                                           position=(10, 3),
                                           magnitude=2.5,
                                           psf_scaling=-1.,
                                           merit='poisson',
                                           aperture=0.06,
                                           sigma=0.,
                                           tolerance=0.1,
                                           pca_number=1,
                                           cent_size=0.06,
                                           edge_size=None,
                                           extra_rot=0.,
                                           reference_in_tag='ref',
                                           residuals='mean')

        self.pipeline.add_module(module)
        self.pipeline.run_module('simplex2')

        data = self.pipeline.get_data('simplex_res_ref')
        assert np.sum(data) == pytest.approx(6.054465042927513, rel=self.limit, abs=0.)
        assert data.shape == (28, 21, 21)

        data = self.pipeline.get_data('flux_position_ref')
        assert data[25, 0] == pytest.approx(9.975388289810159, rel=self.limit, abs=0.)
        assert data[25, 1] == pytest.approx(2.6252170601957676, rel=self.limit, abs=0.)
        assert data[25, 2] == pytest.approx(0.19912024820965318, rel=self.limit, abs=0.)
        assert data[25, 3] == pytest.approx(179.80878869291712, rel=self.limit, abs=0.)
        assert data[25, 4] == pytest.approx(2.4835815867259954, rel=self.limit, abs=0.)
        assert data.shape == (28, 6)

    # def test_mcmc_sampling(self) -> None:
    #
    #     with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
    #         hdf_file['config'].attrs['CPU'] = 4
    #
    #     self.pipeline.set_attribute('adi', 'PARANG', np.arange(0., 200., 10.), static=False)
    #
    #     module = DerotateAndStackModule(name_in='stack',
    #                                     image_in_tag='psf',
    #                                     image_out_tag='psf_stack',
    #                                     derotate=False,
    #                                     stack='mean')
    #
    #     self.pipeline.add_module(module)
    #     self.pipeline.run_module('stack')
    #
    #     data = self.pipeline.get_data('psf_stack')
    #     assert data.shape == (1, 15, 15)
    #
    #     module = MCMCsamplingModule(name_in='mcmc',
    #                                 image_in_tag='adi',
    #                                 psf_in_tag='psf_stack',
    #                                 chain_out_tag='mcmc',
    #                                 param=(0.15, 0., 1.),
    #                                 bounds=((0.1, 0.2), (-2., 2.), (-1., 2.)),
    #                                 nwalkers=50,
    #                                 nsteps=150,
    #                                 psf_scaling=-1.,
    #                                 pca_number=1,
    #                                 aperture=(7, 13, 0.1),
    #                                 mask=None,
    #                                 extra_rot=0.,
    #                                 merit='gaussian',
    #                                 residuals='median',
    #                                 scale=2.,
    #                                 sigma=(1e-3, 1e-1, 1e-2))
    #
    #     self.pipeline.add_module(module)
    #     self.pipeline.run_module('mcmc')
    #
    #     data = self.pipeline.get_data('mcmc')
    #     data = data[50:, :, :].reshape((-1, 3))
    #     assert np.allclose(np.median(data[:, 0]), 0.15, rel=0., abs=0.1)
    #     assert np.allclose(np.median(data[:, 1]), 0., rel=0., abs=1.0)
    #     assert np.allclose(np.median(data[:, 2]), 0.0, rel=0., abs=1.)
    #
    #     attr = self.pipeline.get_attribute('mcmc', 'ACCEPTANCE', static=True)
    #     assert np.allclose(attr, 0.3, rel=0., abs=0.3)
    #
    # def test_systematic_error(self) -> None:
    #
    #     module = SystematicErrorModule(name_in='error',
    #                                    image_in_tag='fake',
    #                                    psf_in_tag='read',
    #                                    offset_out_tag='offset',
    #                                    position=(0.5, 90.),
    #                                    magnitude=6.,
    #                                    angles=(0., 180., 2),
    #                                    psf_scaling=1.,
    #                                    merit='gaussian',
    #                                    aperture=0.1,
    #                                    tolerance=0.1,
    #                                    pca_number=10,
    #                                    mask=(None, None),
    #                                    extra_rot=0.,
    #                                    residuals='median')
    #
    #     self.pipeline.add_module(module)
    #     self.pipeline.run_module('error')
    #
    #     data = self.pipeline.get_data('offset')
    #     assert np.allclose(data[0, 0], -0.004066263849143104, rel=self.limit, abs=0.)
    #     assert np.allclose(np.mean(data), -0.0077784357245382725, rel=self.limit, abs=0.)
    #     assert data.shape == (2, 3)
