import os

import h5py
import pytest
import numpy as np

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
        assert np.sum(data) == pytest.approx(11.012854046962481, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        self.pipeline.set_attribute('adi', 'PARANG', np.linspace(0., 180., 10), static=False)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='psf',
                                   input_dir=self.test_dir+'psf')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('psf')
        assert np.sum(data) == pytest.approx(108.43655133957289, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        module = FitsReadingModule(name_in='read3',
                                   image_tag='ref',
                                   input_dir=self.test_dir+'psf')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read3')

        data = self.pipeline.get_data('ref')
        assert np.sum(data) == pytest.approx(108.43655133957289, rel=self.limit, abs=0.)
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
        assert np.sum(data) == pytest.approx(100.80648929590365, rel=self.limit, abs=0.)
        assert data.shape == (10, 1)

        data_multi = self.pipeline.get_data('photometry2')
        assert data.shape == data_multi.shape
        assert data == pytest.approx(data_multi, rel=self.limit, abs=0.)

    def test_aperture_photometry_position(self) -> None:

        module = AperturePhotometryModule(name_in='photometry3',
                                          image_in_tag='psf',
                                          phot_out_tag='photometry3',
                                          radius=0.1,
                                          position=(10., 10.))

        self.pipeline.add_module(module)
        self.pipeline.run_module('photometry3')

        data = self.pipeline.get_data('photometry3')
        assert np.sum(data) == pytest.approx(100.80648929590365, rel=self.limit, abs=0.)
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
        assert np.sum(data) == pytest.approx(21.273233520675586, rel=self.limit, abs=0.)
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
        assert np.sum(data) == pytest.approx(0.013659056187572433, rel=self.limit, abs=0.)
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
        assert data[0, 4] == pytest.approx(26.661611583224417, rel=self.limit, abs=0.)
        assert data[0, 5] == pytest.approx(1.3375156927387672e-08, rel=self.limit, abs=0.)
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
        assert data[0, 1] == pytest.approx(2.0747802734374985, rel=self.limit, abs=0.)
        assert data[0, 2] == pytest.approx(0.2139883890923524, rel=self.limit, abs=0.)
        assert data[0, 3] == pytest.approx(179.52168877335356, rel=self.limit, abs=0.)
        assert data[0, 4] == pytest.approx(27.457328210661814, rel=self.limit, abs=0.)
        assert data[0, 5] == pytest.approx(1.0905578015907869e-08, rel=self.limit, abs=0.)
        assert data.shape == (1, 6)

    def test_simplex_minimization_hessian(self) -> None:

        module = SimplexMinimizationModule(name_in='simplex1',
                                           image_in_tag='fake',
                                           psf_in_tag='psf',
                                           res_out_tag='simplex_res',
                                           flux_position_tag='flux_position',
                                           position=(10., 3.),
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
                                           offset=1.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('simplex1')

        data = self.pipeline.get_data('simplex_res')

        assert np.sum(data) == pytest.approx(0.2781582369128238, rel=self.limit, abs=0.)
        assert data.shape == (35, 21, 21)

        data = self.pipeline.get_data('flux_position')
        assert data[24, 0] == pytest.approx(9.931627229080938, rel=self.limit, abs=0.)
        assert data[24, 1] == pytest.approx(2.6575231481481456, rel=self.limit, abs=0.)
        assert data[24, 2] == pytest.approx(0.1982554700445013, rel=self.limit, abs=0.)
        assert data[24, 3] == pytest.approx(179.46648003649148, rel=self.limit, abs=0.)
        assert data[24, 4] == pytest.approx(2.5256451474622708, rel=self.limit, abs=0.)
        assert data.shape == (35, 6)

    def test_simplex_minimization_reference(self) -> None:

        module = SimplexMinimizationModule(name_in='simplex2',
                                           image_in_tag='fake',
                                           psf_in_tag='psf',
                                           res_out_tag='simplex_res_ref',
                                           flux_position_tag='flux_position_ref',
                                           position=(10., 3.),
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
        assert np.sum(data) == pytest.approx(9.734993454838076, rel=self.limit, abs=0.)
        assert data.shape == (28, 21, 21)

        data = self.pipeline.get_data('flux_position_ref')
        assert data[27, 0] == pytest.approx(10.049019964116436, rel=self.limit, abs=0.)
        assert data[27, 1] == pytest.approx(2.6444836362361936, rel=self.limit, abs=0.)
        assert data[27, 2] == pytest.approx(0.19860335205689572, rel=self.limit, abs=0.)
        assert data[27, 3] == pytest.approx(180.38183525629643, rel=self.limit, abs=0.)
        assert data[27, 4] == pytest.approx(2.5496922175196, rel=self.limit, abs=0.)
        assert data.shape == (28, 6)

    def test_mcmc_sampling(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        module = DerotateAndStackModule(name_in='stack',
                                        image_in_tag='psf',
                                        image_out_tag='psf_stack',
                                        derotate=False,
                                        stack='mean')

        self.pipeline.add_module(module)
        self.pipeline.run_module('stack')

        data = self.pipeline.get_data('psf_stack')
        assert np.sum(data) == pytest.approx(10.843655133957288, rel=self.limit, abs=0.)
        assert data.shape == (1, 21, 21)

        module = MCMCsamplingModule(name_in='mcmc1',
                                    image_in_tag='adi',
                                    psf_in_tag='psf_stack',
                                    chain_out_tag='mcmc',
                                    param=(0.15, 0., 1.),
                                    bounds=((0.1, 0.2), (-2., 2.), (-1., 2.)),
                                    nwalkers=6,
                                    nsteps=5,
                                    psf_scaling=-1.,
                                    pca_number=1,
                                    aperture=(10, 16, 0.06),
                                    mask=None,
                                    extra_rot=0.,
                                    merit='gaussian',
                                    residuals='median',
                                    resume=False,
                                    sigma=(1e-3, 1e-1, 1e-2))

        self.pipeline.add_module(module)
        self.pipeline.run_module('mcmc1')

        data = self.pipeline.get_data('mcmc')
        assert data.shape == (5, 6, 3)

        data = self.pipeline.get_data('mcmc_backend')
        assert data.shape == (3, )

        module = MCMCsamplingModule(name_in='mcmc2',
                                    image_in_tag='adi',
                                    psf_in_tag='psf_stack',
                                    chain_out_tag='mcmc',
                                    param=(0.15, 0., 1.),
                                    bounds=((0.1, 0.2), (-2., 2.), (-1., 2.)),
                                    nwalkers=6,
                                    nsteps=5,
                                    psf_scaling=-1.,
                                    pca_number=1,
                                    aperture=(10, 16, 0.06),
                                    mask=None,
                                    extra_rot=0.,
                                    merit='gaussian',
                                    residuals='median',
                                    resume=True,
                                    sigma=(1e-3, 1e-1, 1e-2))

        self.pipeline.add_module(module)
        self.pipeline.run_module('mcmc2')

        data = self.pipeline.get_data('mcmc')
        assert data.shape == (10, 6, 3)

        data = self.pipeline.get_data('mcmc_backend')
        assert data.shape == (3, )

    def test_systematic_error(self) -> None:

        module = SystematicErrorModule(name_in='error',
                                       image_in_tag='adi',
                                       psf_in_tag='psf',
                                       offset_out_tag='offset',
                                       position=(0.162, 0.),
                                       magnitude=5.,
                                       angles=(0., 180., 2),
                                       psf_scaling=1.,
                                       merit='gaussian',
                                       aperture=0.06,
                                       tolerance=0.1,
                                       pca_number=1,
                                       mask=(None, None),
                                       extra_rot=0.,
                                       residuals='median',
                                       offset=1.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('error')

        data = self.pipeline.get_data('offset')
        assert data[0, 0] == pytest.approx(-0.001114020093541973, rel=self.limit, abs=0.)
        assert data[0, 1] == pytest.approx(-0.012163271644183737, rel=self.limit, abs=0.)
        assert data[0, 2] == pytest.approx(-0.017943854263249293, rel=self.limit, abs=0.)
        assert data[0, 3] == pytest.approx(0.001282493868968615, rel=self.limit, abs=0.)
        assert data[0, 4] == pytest.approx(-0.04125986733475884, rel=self.limit, abs=0.)
        assert data.shape == (2, 5)
