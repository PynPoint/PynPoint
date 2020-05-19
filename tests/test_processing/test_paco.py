import os

import pytest
import numpy as np

from astropy.io import fits

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.pacoModule import PACOModule
from pynpoint.processing.stacksubset import DerotateAndStackModule
from pynpoint.util.tests import create_config, create_fake_data, create_star_data, remove_test_data


class TestPaco:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_fake_data(self.test_dir+'science')
        create_star_data(self.test_dir+'psf', npix=21, pos_star=10.)
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['science', 'psf'])

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
                                   image_tag='psf',
                                   input_dir=self.test_dir+'psf')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('psf')
        assert np.sum(data) == pytest.approx(108.43655133957289, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

    def test_stack_psf(self) -> None:

        module = DerotateAndStackModule(name_in='stack',
                                        image_in_tag='psf',
                                        image_out_tag='psf_stack',
                                        derotate=False,
                                        stack='mean',
                                        extra_rot=0.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('stack')

        data = self.pipeline.get_data('psf_stack')
        assert np.sum(data) == pytest.approx(10.843655133957288, rel=self.limit, abs=0.)
        assert data.shape == (1, 21, 21)

    def test_paco_fast(self) -> None:

        module = PACOModule(name_in='paco',
                            image_in_tag='science',
                            psf_in_tag='psf_stack',
                            snr_out_tag='snr',
                            flux_out_tag='flux',
                            psf_rad=3.,
                            scaling=0.1,
                            algorithm='fastpaco',
                            flux_calc=True,
                            threshold=5.,
                            flux_prec=0.05)

        self.pipeline.add_module(module)

        with pytest.warns(None) as warning:
            self.pipeline.run_module('paco')

        assert len(warning) == 3

        data = self.pipeline.get_data('snr')
        # fits.writeto('data.fits', data, overwrite=True)
        # assert np.sum(data) == pytest.approx(5.029285028467547, rel=self.limit, abs=0.)
        assert data.shape == (2, 2)

        data = self.pipeline.get_data('flux')
        # assert np.sum(data) == pytest.approx(5.029285028467547, rel=self.limit, abs=0.)
        assert data.shape == (2, 2)
