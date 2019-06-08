import os
import warnings

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.fluxposition import FakePlanetModule, AperturePhotometryModule, \
                                             FalsePositiveModule, SimplexMinimizationModule, \
                                             MCMCsamplingModule
from pynpoint.processing.resizing import ScaleImagesModule
from pynpoint.processing.stacksubset import DerotateAndStackModule
from pynpoint.processing.psfpreparation import AngleInterpolationModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule
from pynpoint.util.tests import create_config, create_star_data, create_fake, remove_test_data

warnings.simplefilter('always')

limit = 1e-10

class TestFluxPosition:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(path=self.test_dir+'flux', npix_x=101, npix_y=101)

        create_star_data(path=self.test_dir+'ref', npix_x=101, npix_y=101)

        create_star_data(path=self.test_dir+'psf',
                         npix_x=15,
                         npix_y=15,
                         x0=[7., 7., 7., 7.],
                         y0=[7., 7., 7., 7.],
                         ndit=1,
                         nframes=1,
                         noise=False)

        create_fake(path=self.test_dir+'adi',
                    ndit=[5, 5, 5, 5],
                    nframes=[5, 5, 5, 5],
                    exp_no=[1, 2, 3, 4],
                    npix=(15, 15),
                    fwhm=3.,
                    x0=[7., 7., 7., 7.],
                    y0=[7., 7., 7., 7.],
                    angles=[[0., 50.], [50., 100.], [100., 150.], [150., 200.]],
                    sep=5.5,
                    contrast=1.)

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['flux', 'adi', 'psf', 'ref'])

    def test_read_data(self):

        module = FitsReadingModule(name_in='read1',
                                   image_tag='read',
                                   input_dir=self.test_dir+'flux')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        data = self.pipeline.get_data('read')
        assert np.allclose(data[0, 50, 50], 0.0986064357966972, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.827812356946396e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 101, 101)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='adi',
                                   input_dir=self.test_dir+'adi')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('adi')
        assert np.allclose(data[0, 7, 7], 0.09823888178122618, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.008761678820997612, rtol=limit, atol=0.)
        assert data.shape == (20, 15, 15)

        module = FitsReadingModule(name_in='read3',
                                   image_tag='psf',
                                   input_dir=self.test_dir+'psf')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read3')

        data = self.pipeline.get_data('psf')
        assert np.allclose(data[0, 7, 7], 0.09806026673451182, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.004444444429123135, rtol=limit, atol=0.)
        assert data.shape == (4, 15, 15)

        module = FitsReadingModule(name_in='read4',
                                   image_tag='ref',
                                   input_dir=self.test_dir+'ref')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read4')

        data = self.pipeline.get_data('ref')
        assert np.allclose(data[0, 50, 50], 0.0986064357966972, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.827812356946396e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 101, 101)

    def test_aperture_photometry(self):

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 1

        module = AperturePhotometryModule(radius=0.1,
                                          position=None,
                                          name_in='photometry',
                                          image_in_tag='read',
                                          phot_out_tag='photometry')

        self.pipeline.add_module(module)
        self.pipeline.run_module('photometry')

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        module = AperturePhotometryModule(radius=0.1,
                                          position=None,
                                          name_in='photometry_multi',
                                          image_in_tag='read',
                                          phot_out_tag='photometry_multi')

        self.pipeline.add_module(module)
        self.pipeline.run_module('photometry_multi')

        data = self.pipeline.get_data('photometry')
        assert np.allclose(data[0][0], 0.9853286992326858, rtol=limit, atol=0.)
        assert np.allclose(data[39][0], 0.9835251375574492, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.9836439188900222, rtol=limit, atol=0.)
        assert data.shape == (40, 1)

        data_multi = self.pipeline.get_data('photometry_multi')
        assert data.shape == data_multi.shape

        # Outputs zeros sometimes for data_multi on Travis CI
        # for i, item in enumerate(data_multi):
        #     assert np.allclose(data[i], item, rtol=1e-6, atol=0.)

    def test_angle_interpolation(self):

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 1

        module = AngleInterpolationModule(name_in='angle',
                                          data_tag='read')

        self.pipeline.add_module(module)
        self.pipeline.run_module('angle')

        data = self.pipeline.get_data('header_read/PARANG')
        assert data[5] == 2.7777777777777777
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_fake_planet(self):

        module = FakePlanetModule(position=(0.5, 90.),
                                  magnitude=6.,
                                  psf_scaling=1.,
                                  interpolation='spline',
                                  name_in='fake',
                                  image_in_tag='read',
                                  psf_in_tag='read',
                                  image_out_tag='fake')

        self.pipeline.add_module(module)
        self.pipeline.run_module('fake')

        data = self.pipeline.get_data('fake')
        assert np.allclose(data[0, 50, 50], 0.09860622347589054, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.867026482551375e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 101, 101)

    def test_psf_subtraction(self):

        module = PcaPsfSubtractionModule(pca_numbers=(2, ),
                                         name_in='pca',
                                         images_in_tag='fake',
                                         reference_in_tag='fake',
                                         res_mean_tag='res_mean',
                                         res_median_tag=None,
                                         res_arr_out_tag=None,
                                         res_rot_mean_clip_tag=None,
                                         extra_rot=0.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca')

        data = self.pipeline.get_data('res_mean')
        assert np.allclose(data[0, 49, 31], 4.8963214463463886e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.8409659677297164e-08, rtol=limit, atol=0.)
        assert data.shape == (1, 101, 101)

    def test_false_positive(self):

        module = FalsePositiveModule(position=(31., 49.),
                                     aperture=0.1,
                                     ignore=True,
                                     name_in='false',
                                     image_in_tag='res_mean',
                                     snr_out_tag='snr_fpf')

        self.pipeline.add_module(module)
        self.pipeline.run_module('false')

        data = self.pipeline.get_data('snr_fpf')
        assert np.allclose(data[0, 0], 31.0, rtol=limit, atol=0.)
        assert np.allclose(data[0, 1], 49.0, rtol=limit, atol=0.)
        assert np.allclose(data[0, 2], 0.513710034941892, rtol=limit, atol=0.)
        assert np.allclose(data[0, 3], 93.01278750418334, rtol=limit, atol=0.)
        assert np.allclose(data[0, 4], 7.333740467578795, rtol=limit, atol=0.)
        assert np.allclose(data[0, 5], 4.5257622875993775e-06, rtol=limit, atol=0.)

    def test_simplex_minimization(self):

        module = SimplexMinimizationModule(name_in='simplex1',
                                           image_in_tag='fake',
                                           psf_in_tag='read',
                                           res_out_tag='simplex_res',
                                           flux_position_tag='flux_position',
                                           position=(31., 49.),
                                           magnitude=6.,
                                           psf_scaling=-1.,
                                           merit='hessian',
                                           aperture=0.1,
                                           sigma=0.,
                                           tolerance=0.1,
                                           pca_number=1,
                                           cent_size=0.1,
                                           edge_size=None,
                                           extra_rot=0.,
                                           reference_in_tag=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('simplex1')

        data = self.pipeline.get_data('simplex_res')
        assert np.allclose(data[0, 50, 31], 0.00012976212788352575, rtol=limit, atol=0.)
        assert np.allclose(data[42, 50, 31], 1.2141761821389107e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.461337432531517e-09, rtol=limit, atol=0.)
        assert data.shape == (43, 101, 101)

        data = self.pipeline.get_data('flux_position')
        assert np.allclose(data[42, 0], 31.6456737445356, rtol=limit, atol=0.)
        assert np.allclose(data[42, 1], 49.9199601480223, rtol=limit, atol=0.)
        assert np.allclose(data[42, 2], 0.49557152090327206, rtol=limit, atol=0.)
        assert np.allclose(data[42, 3], 90.24985480686087, rtol=limit, atol=0.)
        assert np.allclose(data[42, 4], 5.683191873535635, rtol=limit, atol=0.)
        assert data.shape == (43, 6)

    def test_simplex_minimization_reference(self):

        module = SimplexMinimizationModule(name_in='simplex2',
                                           image_in_tag='fake',
                                           psf_in_tag='read',
                                           res_out_tag='simplex_res_ref',
                                           flux_position_tag='flux_position_ref',
                                           position=(31., 49.),
                                           magnitude=6.,
                                           psf_scaling=-1.,
                                           merit='sum',
                                           aperture=0.1,
                                           sigma=0.,
                                           tolerance=0.1,
                                           pca_number=1,
                                           cent_size=0.1,
                                           edge_size=None,
                                           extra_rot=0.,
                                           reference_in_tag='ref')

        self.pipeline.add_module(module)
        self.pipeline.run_module('simplex2')

        data = self.pipeline.get_data('simplex_res_ref')
        assert np.allclose(data[0, 50, 31], 0.00014188043631450017, rtol=limit, atol=0.)
        assert np.allclose(data[35, 50, 31], 8.260705332688204e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.3084672332295342e-06, rtol=limit, atol=0.)
        assert data.shape == (36, 101, 101)

        data = self.pipeline.get_data('flux_position_ref')
        assert np.allclose(data[35, 0], 31.523599108957953, rtol=limit, atol=0.)
        assert np.allclose(data[35, 1], 49.80956476439131, rtol=limit, atol=0.)
        assert np.allclose(data[35, 2], 0.49888932122698404, rtol=limit, atol=0.)
        assert np.allclose(data[35, 3], 90.59052349996864, rtol=limit, atol=0.)
        assert np.allclose(data[35, 4], 6.021039673141669, rtol=limit, atol=0.)
        assert data.shape == (36, 6)

    def test_mcmc_sampling_gaussian(self):

        self.pipeline.set_attribute('adi', 'PARANG', np.arange(0., 200., 10.), static=False)

        module = ScaleImagesModule(scaling=(None, None, 100.),
                                   pixscale=False,
                                   name_in='scale1',
                                   image_in_tag='adi',
                                   image_out_tag='adi_scale')


        self.pipeline.add_module(module)
        self.pipeline.run_module('scale1')

        data = self.pipeline.get_data('adi_scale')
        assert np.allclose(data[0, 7, 7], 9.82388817812263, rtol=limit, atol=0.)
        assert data.shape == (20, 15, 15)

        module = ScaleImagesModule(scaling=(None, None, 100.),
                                   pixscale=False,
                                   name_in='scale2',
                                   image_in_tag='psf',
                                   image_out_tag='psf_scale')


        self.pipeline.add_module(module)
        self.pipeline.run_module('scale2')

        data = self.pipeline.get_data('psf_scale')
        assert np.allclose(data[0, 7, 7], 9.806026673451198, rtol=limit, atol=0.)
        assert data.shape == (4, 15, 15)

        module = DerotateAndStackModule(name_in='take_psf_avg',
                                        image_in_tag='psf_scale',
                                        image_out_tag='psf_avg',
                                        derotate=False,
                                        stack='mean')

        self.pipeline.add_module(module)
        self.pipeline.run_module('take_psf_avg')

        data = self.pipeline.get_data('psf_avg')
        assert data.shape == (1, 15, 15)

        module = MCMCsamplingModule(param=(0.1485, 0., 0.),
                                    bounds=((0.1, 0.25), (-5., 5.), (-0.5, 0.5)),
                                    name_in='mcmc',
                                    image_in_tag='adi_scale',
                                    psf_in_tag='psf_avg',
                                    chain_out_tag='mcmc',
                                    nwalkers=50,
                                    nsteps=150,
                                    psf_scaling=-1.,
                                    pca_number=1,
                                    aperture={'type':'circular',
                                              'pos_x':7.0,
                                              'pos_y':12.5,
                                              'radius':0.1},
                                    mask=None,
                                    extra_rot=0.,
                                    scale=2.,
                                    sigma=(1e-3, 1e-1, 1e-2),
                                    prior='flat',
                                    variance='gaussian')

        self.pipeline.add_module(module)

        with pytest.warns(FutureWarning) as warning:
            self.pipeline.run_module('mcmc')

        assert warning[0].message.args[0] == 'Using a non-tuple sequence for multidimensional ' \
                                             'indexing is deprecated; use `arr[tuple(seq)]` ' \
                                             'instead of `arr[seq]`. In the future this will be ' \
                                             'interpreted as an array index, ' \
                                             '`arr[np.array(seq)]`, which will result either ' \
                                             'in an error or a different result.'

        single = self.pipeline.get_data('mcmc')
        single = single[:, 20:, :].reshape((-1, 3))
        assert np.allclose(np.median(single[:, 0]), 0.148, rtol=0., atol=0.01)
        assert np.allclose(np.median(single[:, 1]), 0., rtol=0., atol=0.2)
        assert np.allclose(np.median(single[:, 2]), 0., rtol=0., atol=0.1)

    def test_mcmc_sampling_poisson(self):

        module = MCMCsamplingModule(param=(0.1485, 0., 0.),
                                    bounds=((0.1, 0.25), (-5., 5.), (-0.5, 0.5)),
                                    name_in='mcmc_prior',
                                    image_in_tag='adi_scale',
                                    psf_in_tag='psf_avg',
                                    chain_out_tag='mcmc_prior',
                                    nwalkers=50,
                                    nsteps=150,
                                    psf_scaling=-1.,
                                    pca_number=1,
                                    aperture={'type':'elliptical',
                                              'pos_x':7.0,
                                              'pos_y':12.5,
                                              'semimajor':0.1,
                                              'semiminor':0.1,
                                              'angle':0.0},
                                    mask=None,
                                    extra_rot=0.,
                                    scale=2.,
                                    sigma=(1e-3, 1e-1, 1e-2),
                                    prior='aperture',
                                    variance='poisson')

        self.pipeline.add_module(module)

        with pytest.warns(FutureWarning) as warning:
            self.pipeline.run_module('mcmc_prior')

        assert warning[0].message.args[0] == 'Using a non-tuple sequence for multidimensional ' \
                                             'indexing is deprecated; use `arr[tuple(seq)]` ' \
                                             'instead of `arr[seq]`. In the future this will be ' \
                                             'interpreted as an array index, ' \
                                             '`arr[np.array(seq)]`, which will result either ' \
                                             'in an error or a different result.'

        single = self.pipeline.get_data('mcmc_prior')
        single = single[:, 20:, :].reshape((-1, 3))
        assert np.allclose(np.median(single[:, 0]), 0.148, rtol=0., atol=0.01)
        assert np.allclose(np.median(single[:, 1]), 0., rtol=0., atol=0.2)
        assert np.allclose(np.median(single[:, 2]), 0., rtol=0., atol=0.1)

    def test_mcmc_sampling_wrong_prior(self):

        module = MCMCsamplingModule(param=(0.1485, 0., 0.),
                                    bounds=((0.1, 0.25), (-5., 5.), (-0.5, 0.5)),
                                    name_in='mcmc_wrong_prior',
                                    image_in_tag='adi_scale',
                                    psf_in_tag='psf_avg',
                                    chain_out_tag='mcmc_prior',
                                    nwalkers=50,
                                    nsteps=150,
                                    psf_scaling=-1.,
                                    pca_number=1,
                                    aperture={'type':'circular',
                                              'pos_x':7.0,
                                              'pos_y':12.5,
                                              'radius':0.1},
                                    mask=None,
                                    extra_rot=0.,
                                    scale=2.,
                                    sigma=(1e-3, 1e-1, 1e-2),
                                    prior='test',
                                    variance='gaussian')

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('mcmc_wrong_prior')

        assert str(error.value) == 'Prior type not recognized.'
