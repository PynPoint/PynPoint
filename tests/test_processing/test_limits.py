import os

from urllib.request import urlretrieve

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.limits import ContrastCurveModule, MassLimitsModule
from pynpoint.processing.psfpreparation import AngleInterpolationModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data


class TestLimits:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(self.test_dir+'self.limits', npix=21, pos_star=10.)
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(path=self.test_dir,
                         folders=['self.limits'],
                         files=['model.AMES-Cond-2000.M-0.0.NaCo.Vega'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read',
                                   image_tag='read',
                                   input_dir=self.test_dir+'self.limits')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read')

        data = self.pipeline.get_data('read')
        assert np.sum(data) == pytest.approx(108.43655133957289, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

    def test_angle_interpolation(self) -> None:

        module = AngleInterpolationModule(name_in='angle',
                                          data_tag='read')

        self.pipeline.add_module(module)
        self.pipeline.run_module('angle')

        attr = self.pipeline.get_attribute('read', 'PARANG', static=False)
        assert np.sum(attr) == pytest.approx(900., rel=self.limit, abs=0.)
        assert attr.shape == (10, )

    def test_contrast_curve(self) -> None:

        proc = ['single', 'multi']

        for item in proc:

            if item == 'multi':
                with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
                    hdf_file['config'].attrs['CPU'] = 4

            module = ContrastCurveModule(name_in='contrast_'+item,
                                         image_in_tag='read',
                                         psf_in_tag='read',
                                         contrast_out_tag='limits_'+item,
                                         separation=(0.2, 0.3, 0.2),
                                         angle=(0., 360., 180.),
                                         threshold=('sigma', 5.),
                                         psf_scaling=1.,
                                         aperture=0.05,
                                         pca_number=2,
                                         cent_size=None,
                                         edge_size=1.,
                                         extra_rot=0.)

            self.pipeline.add_module(module)
            self.pipeline.run_module('contrast_'+item)

            data = self.pipeline.get_data('limits_'+item)
            assert data[0, 0] == pytest.approx(0.2, rel=self.limit, abs=0.)
            assert data[0, 1] == pytest.approx(2.5223717329932676, rel=self.limit, abs=0.)
            assert data[0, 2] == pytest.approx(0.0006250749411563979, rel=self.limit, abs=0.)
            assert data[0, 3] == pytest.approx(0.00026866680137822624, rel=self.limit, abs=0.)
            assert data.shape == (1, 4)

    def test_contrast_curve_fpf(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 1

        module = ContrastCurveModule(name_in='contrast_fpf',
                                     image_in_tag='read',
                                     psf_in_tag='read',
                                     contrast_out_tag='limits_fpf',
                                     separation=(0.2, 0.3, 0.2),
                                     angle=(0., 360., 180.),
                                     threshold=('fpf', 1e-6),
                                     psf_scaling=1.,
                                     aperture=0.05,
                                     pca_number=2,
                                     cent_size=None,
                                     edge_size=1.,
                                     extra_rot=0.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('contrast_fpf')

        data = self.pipeline.get_data('limits_fpf')
        assert data[0, 0] == pytest.approx(0.2, rel=self.limit, abs=0.)
        assert data[0, 1] == pytest.approx(1.797063014325614, rel=self.limit, abs=0.)
        assert data[0, 2] == pytest.approx(0.0006250749411564145, rel=self.limit, abs=0.)
        assert data[0, 3] == pytest.approx(1e-06, rel=self.limit, abs=0.)
        assert data.shape == (1, 4)

    def test_mass_limits(self) -> None:

        separation = np.linspace(0.1, 1.0, 10)
        contrast = -2.5*np.log10(1e-4/separation)
        variance = 0.1*contrast

        limits = np.zeros((10, 4))
        limits[:, 0] = separation
        limits[:, 1] = contrast
        limits[:, 2] = variance

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['contrast_limits'] = limits

        url = 'https://phoenix.ens-lyon.fr/Grids/AMES-Cond/ISOCHRONES/' \
              'model.AMES-Cond-2000.M-0.0.NaCo.Vega'

        filename = self.test_dir + 'model.AMES-Cond-2000.M-0.0.NaCo.Vega'

        urlretrieve(url, filename)

        module = MassLimitsModule(model_file=filename,
                                  star_prop={'magnitude': 10., 'distance': 100., 'age': 20.},
                                  name_in='mass',
                                  contrast_in_tag='contrast_limits',
                                  mass_out_tag='mass_limits',
                                  instr_filter='L\'')

        self.pipeline.add_module(module)
        self.pipeline.run_module('mass')

        data = self.pipeline.get_data('mass_limits')
        assert np.mean(data[:, 0]) == pytest.approx(0.55, rel=self.limit, abs=0.)
        assert np.mean(data[:, 1]) == pytest.approx(0.001891690765603738, rel=self.limit, abs=0.)
        assert np.mean(data[:, 2]) == pytest.approx(0.000964309686441908, rel=self.limit, abs=0.)
        assert np.mean(data[:, 3]) == pytest.approx(-0.000696402843279597, rel=self.limit, abs=0.)
        assert data.shape == (10, 4)
