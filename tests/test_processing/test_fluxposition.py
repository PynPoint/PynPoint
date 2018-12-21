import os
import warnings

import h5py
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

warnings.simplefilter("always")

limit = 1e-10

class TestFluxAndPosition(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"flux", npix_x=101, npix_y=101)

        create_star_data(path=self.test_dir+"psf",
                         npix_x=15,
                         npix_y=15,
                         x0=[7., 7., 7., 7.],
                         y0=[7., 7., 7., 7.],
                         ndit=1,
                         nframes=1,
                         noise=False)

        create_fake(path=self.test_dir+"adi",
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

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["flux", "adi", "psf"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read1",
                                 image_tag="read",
                                 input_dir=self.test_dir+"flux")

        self.pipeline.add_module(read)
        self.pipeline.run_module("read1")

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 50, 50], 0.0986064357966972, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.827812356946396e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 101, 101)

        read = FitsReadingModule(name_in="read2",
                                 image_tag="adi",
                                 input_dir=self.test_dir+"adi")

        self.pipeline.add_module(read)
        self.pipeline.run_module("read2")

        data = self.pipeline.get_data("adi")
        assert np.allclose(data[0, 7, 7], 0.09823888178122618, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.008761678820997612, rtol=limit, atol=0.)
        assert data.shape == (20, 15, 15)

        read = FitsReadingModule(name_in="read3",
                                 image_tag="psf",
                                 input_dir=self.test_dir+"psf")

        self.pipeline.add_module(read)
        self.pipeline.run_module("read3")

        data = self.pipeline.get_data("psf")
        assert np.allclose(data[0, 7, 7], 0.09806026673451182, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.004444444429123135, rtol=limit, atol=0.)
        assert data.shape == (4, 15, 15)

    def test_aperture_photometry(self):

        photometry = AperturePhotometryModule(radius=0.1,
                                              position=None,
                                              name_in="photometry",
                                              image_in_tag="read",
                                              phot_out_tag="photometry")

        self.pipeline.add_module(photometry)
        self.pipeline.run_module("photometry")

        data = self.pipeline.get_data("photometry")
        assert np.allclose(data[0][0], 0.9702137183213615, rtol=limit, atol=0.)
        assert np.allclose(data[39][0], 0.9691512171281103, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.9691752104364761, rtol=limit, atol=0.)
        assert data.shape == (40, 1)

    def test_angle_interpolation(self):

        angle = AngleInterpolationModule(name_in="angle",
                                         data_tag="read")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle")

        data = self.pipeline.get_data("header_read/PARANG")
        assert data[5] == 2.7777777777777777
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_fake_planet(self):

        fake = FakePlanetModule(position=(0.5, 90.),
                                magnitude=6.,
                                psf_scaling=1.,
                                interpolation="spline",
                                name_in="fake",
                                image_in_tag="read",
                                psf_in_tag="read",
                                image_out_tag="fake")

        self.pipeline.add_module(fake)
        self.pipeline.run_module("fake")

        data = self.pipeline.get_data("fake")
        assert np.allclose(data[0, 50, 50], 0.09860622347589054, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.867026482551375e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 101, 101)

    def test_psf_subtraction(self):

        pca = PcaPsfSubtractionModule(pca_numbers=(2, ),
                                      name_in="pca",
                                      images_in_tag="fake",
                                      reference_in_tag="fake",
                                      res_mean_tag="res_mean",
                                      res_median_tag=None,
                                      res_arr_out_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      extra_rot=0.)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca")

        data = self.pipeline.get_data("res_mean")
        assert np.allclose(data[0, 49, 31], 4.8963214463463886e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.8409659677297164e-08, rtol=limit, atol=0.)
        assert data.shape == (1, 101, 101)

    def test_false_positive(self):

        false = FalsePositiveModule(position=(31., 49.),
                                    aperture=0.1,
                                    ignore=True,
                                    name_in="false",
                                    image_in_tag="res_mean",
                                    snr_out_tag="snr_fpf")

        self.pipeline.add_module(false)
        self.pipeline.run_module("false")

        data = self.pipeline.get_data("snr_fpf")
        assert np.allclose(data[0, 2], 0.5280553948214145, rtol=limit, atol=0.)
        assert np.allclose(data[0, 3], 94.39870535499551, rtol=limit, atol=0.)
        assert np.allclose(data[0, 4], 8.542166952478182, rtol=limit, atol=0.)
        assert np.allclose(data[0, 5], 9.54772666372783e-07, rtol=limit, atol=0.)

    def test_simplex_minimization(self):

        simplex = SimplexMinimizationModule(position=(31., 49.),
                                            magnitude=6.,
                                            psf_scaling=-1.,
                                            name_in="simplex",
                                            image_in_tag="fake",
                                            psf_in_tag="read",
                                            res_out_tag="simplex_res",
                                            flux_position_tag="flux_position",
                                            merit="hessian",
                                            aperture=0.1,
                                            sigma=0.,
                                            tolerance=0.1,
                                            pca_number=1,
                                            cent_size=0.1,
                                            edge_size=None,
                                            extra_rot=0.)

        self.pipeline.add_module(simplex)
        self.pipeline.run_module("simplex")

        data = self.pipeline.get_data("simplex_res")
        assert np.allclose(data[0, 50, 31], 0.00020085220731657478, rtol=limit, atol=0.)
        assert np.allclose(data[65, 50, 31], 2.5035345163849688e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.416893585673146e-09, rtol=limit, atol=0.)
        assert data.shape == (66, 101, 101)

        data = self.pipeline.get_data("flux_position")
        assert np.allclose(data[65, 0], 32.14539423594633, rtol=limit, atol=0.)
        assert np.allclose(data[65, 1], 50.40994810153265, rtol=limit, atol=0.)
        assert np.allclose(data[65, 2], 0.4955803200991986, rtol=limit, atol=0.)
        assert np.allclose(data[65, 3], 90.28110395762462, rtol=limit, atol=0.)
        assert np.allclose(data[65, 4], 5.744096115502183, rtol=limit, atol=0.)
        assert data.shape == (66, 6)

    def test_mcmc_sampling(self):

        self.pipeline.set_attribute("adi", "PARANG", np.arange(0., 200., 10.), static=False)

        scale = ScaleImagesModule(scaling=(None, None, 100.),
                                  pixscale=False,
                                  name_in="scale1",
                                  image_in_tag="adi",
                                  image_out_tag="adi_scale")


        self.pipeline.add_module(scale)
        self.pipeline.run_module("scale1")

        data = self.pipeline.get_data("adi_scale")
        assert np.allclose(data[0, 7, 7], 9.82388817812263, rtol=limit, atol=0.)
        assert data.shape == (20, 15, 15)

        scale = ScaleImagesModule(scaling=(None, None, 100.),
                                  pixscale=False,
                                  name_in="scale2",
                                  image_in_tag="psf",
                                  image_out_tag="psf_scale")


        self.pipeline.add_module(scale)
        self.pipeline.run_module("scale2")

        data = self.pipeline.get_data("psf_scale")
        assert np.allclose(data[0, 7, 7], 9.806026673451198, rtol=limit, atol=0.)
        assert data.shape == (4, 15, 15)

        avg_psf = DerotateAndStackModule(name_in="take_psf_avg",
                                         image_in_tag="psf_scale",
                                         image_out_tag="psf_avg",
                                         derotate=False,
                                         stack="mean")

        self.pipeline.add_module(avg_psf)
        self.pipeline.run_module("take_psf_avg")

        data = self.pipeline.get_data("psf_avg")
        assert data.shape == (15, 15)

        mcmc = MCMCsamplingModule(param=(0.1485, 0., 0.),
                                  bounds=((0.1, 0.25), (-5., 5.), (-0.5, 0.5)),
                                  name_in="mcmc",
                                  image_in_tag="adi_scale",
                                  psf_in_tag="psf_avg",
                                  chain_out_tag="mcmc",
                                  nwalkers=50,
                                  nsteps=150,
                                  psf_scaling=-1.,
                                  pca_number=1,
                                  aperture=0.1,
                                  mask=None,
                                  extra_rot=0.,
                                  scale=2.,
                                  sigma=(1e-3, 1e-1, 1e-2),
                                  prior="flat")

        self.pipeline.add_module(mcmc)
        self.pipeline.run_module("mcmc")

        single = self.pipeline.get_data("mcmc")
        single = single[:, 20:, :].reshape((-1, 3))
        assert np.allclose(np.median(single[:, 0]), 0.148, rtol=0., atol=0.01)
        assert np.allclose(np.median(single[:, 1]), 0., rtol=0., atol=0.2)
        assert np.allclose(np.median(single[:, 2]), 0., rtol=0., atol=0.1)

        database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
        database['config'].attrs['CPU'] = 4

        self.pipeline.run_module("mcmc")
        multi = self.pipeline.get_data("mcmc")
        multi = multi[:, 20:, :].reshape((-1, 3))
        assert np.allclose(np.median(multi[:, 0]), 0.148, rtol=0., atol=0.01)
        assert np.allclose(np.median(multi[:, 1]), 0., rtol=0., atol=0.2)
        assert np.allclose(np.median(multi[:, 2]), 0., rtol=0., atol=0.1)
