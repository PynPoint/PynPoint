import os

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.frameselection import RemoveFramesModule, FrameSelectionModule, \
                                               RemoveLastFrameModule, RemoveStartFramesModule, \
                                               ImageStatisticsModule, FrameSimilarityModule, \
                                               SelectByAttributeModule, ResidualSelectionModule
from pynpoint.util.tests import create_config, remove_test_data, create_star_data


class TestFrameSelection:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(self.test_dir+'images')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['images'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read',
                                   image_tag='read',
                                   input_dir=self.test_dir+'images',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read')

        data = self.pipeline.get_data('read')
        assert np.sum(data) == pytest.approx(105.54278879805277, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

        attr = self.pipeline.get_attribute('read', 'NDIT', static=False)
        assert np.sum(attr) == pytest.approx(10, rel=self.limit, abs=0.)
        assert attr.shape == (2, )

        attr = self.pipeline.get_attribute('read', 'NFRAMES', static=False)
        assert np.sum(attr) == pytest.approx(10, rel=self.limit, abs=0.)
        assert attr.shape == (2, )

        self.pipeline.set_attribute('read', 'NDIT', [4, 4], static=False)

    def test_remove_last_frame(self) -> None:

        module = RemoveLastFrameModule(name_in='last',
                                       image_in_tag='read',
                                       image_out_tag='last')

        self.pipeline.add_module(module)
        self.pipeline.run_module('last')

        data = self.pipeline.get_data('last')
        assert np.sum(data) == pytest.approx(84.68885503527224, rel=self.limit, abs=0.)
        assert data.shape == (8, 11, 11)

        self.pipeline.set_attribute('last', 'PARANG', np.arange(8.), static=False)
        self.pipeline.set_attribute('last', 'STAR_POSITION', np.full((8, 2), 5.), static=False)

        attr = self.pipeline.get_attribute('last', 'PARANG', static=False)
        assert np.sum(attr) == pytest.approx(28., rel=self.limit, abs=0.)
        assert attr.shape == (8, )

        attr = self.pipeline.get_attribute('last', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(80., rel=self.limit, abs=0.)
        assert attr.shape == (8, 2)

    def test_remove_start_frame(self) -> None:

        module = RemoveStartFramesModule(frames=1,
                                         name_in='start',
                                         image_in_tag='last',
                                         image_out_tag='start')

        self.pipeline.add_module(module)
        self.pipeline.run_module('start')

        data = self.pipeline.get_data('start')
        assert np.sum(data) == pytest.approx(64.44307047549808, rel=self.limit, abs=0.)
        assert data.shape == (6, 11, 11)

        attr = self.pipeline.get_attribute('start', 'PARANG', static=False)
        assert np.sum(attr) == pytest.approx(24., rel=self.limit, abs=0.)
        assert attr.shape == (6, )

        attr = self.pipeline.get_attribute('start', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(60., rel=self.limit, abs=0.)
        assert attr.shape == (6, 2)

    def test_remove_frames(self) -> None:

        module = RemoveFramesModule(name_in='remove',
                                    image_in_tag='start',
                                    selected_out_tag='selected',
                                    removed_out_tag='removed',
                                    frames=[2, 5])

        self.pipeline.add_module(module)
        self.pipeline.run_module('remove')

        data = self.pipeline.get_data('selected')
        assert np.sum(data) == pytest.approx(43.68337741822863, rel=self.limit, abs=0.)
        assert data.shape == (4, 11, 11)

        data = self.pipeline.get_data('removed')
        assert np.sum(data) == pytest.approx(20.759693057269445, rel=self.limit, abs=0.)
        assert data.shape == (2, 11, 11)

        attr = self.pipeline.get_attribute('selected', 'PARANG', static=False)
        assert np.sum(attr) == pytest.approx(14., rel=self.limit, abs=0.)
        assert attr.shape == (4, )

        attr = self.pipeline.get_attribute('selected', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(40., rel=self.limit, abs=0.)
        assert attr.shape == (4, 2)

        attr = self.pipeline.get_attribute('removed', 'PARANG', static=False)
        assert np.sum(attr) == pytest.approx(10., rel=self.limit, abs=0.)
        assert attr.shape == (2, )

        attr = self.pipeline.get_attribute('removed', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(20., rel=self.limit, abs=0.)
        assert attr.shape == (2, 2)

    def test_frame_selection(self) -> None:

        module = FrameSelectionModule(name_in='select1',
                                      image_in_tag='start',
                                      selected_out_tag='selected1',
                                      removed_out_tag='removed1',
                                      index_out_tag='index1',
                                      method='median',
                                      threshold=2.,
                                      fwhm=0.1,
                                      aperture=('circular', 0.1),
                                      position=(None, None, 0.2))

        self.pipeline.add_module(module)
        self.pipeline.run_module('select1')

        data = self.pipeline.get_data('selected1')
        assert np.sum(data) == pytest.approx(54.58514780071149, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

        data = self.pipeline.get_data('removed1')
        assert np.sum(data) == pytest.approx(9.857922674786586, rel=self.limit, abs=0.)
        assert data.shape == (1, 11, 11)

        data = self.pipeline.get_data('index1')
        assert np.sum(data) == pytest.approx(5, rel=self.limit, abs=0.)
        assert data.shape == (1, )

        attr = self.pipeline.get_attribute('selected1', 'PARANG', static=False)
        assert np.sum(attr) == pytest.approx(17., rel=self.limit, abs=0.)
        assert attr.shape == (5, )

        attr = self.pipeline.get_attribute('selected1', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(50, rel=self.limit, abs=0.)
        assert attr.shape == (5, 2)

        attr = self.pipeline.get_attribute('removed1', 'PARANG', static=False)
        assert np.sum(attr) == pytest.approx(7., rel=self.limit, abs=0.)
        assert attr.shape == (1, )

        attr = self.pipeline.get_attribute('removed1', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(10, rel=self.limit, abs=0.)
        assert attr.shape == (1, 2)

        module = FrameSelectionModule(name_in='select2',
                                      image_in_tag='start',
                                      selected_out_tag='selected2',
                                      removed_out_tag='removed2',
                                      index_out_tag='index2',
                                      method='max',
                                      threshold=1.,
                                      fwhm=0.1,
                                      aperture=('annulus', 0.05, 0.1),
                                      position=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('select2')

        data = self.pipeline.get_data('selected2')
        assert np.sum(data) == pytest.approx(21.42652724866543, rel=self.limit, abs=0.)
        assert data.shape == (2, 11, 11)

        data = self.pipeline.get_data('removed2')
        assert np.sum(data) == pytest.approx(43.016543226832646, rel=self.limit, abs=0.)
        assert data.shape == (4, 11, 11)

        data = self.pipeline.get_data('index2')
        assert np.sum(data) == pytest.approx(10, rel=self.limit, abs=0.)
        assert data.shape == (4, )

        attr = self.pipeline.get_attribute('selected2', 'PARANG', static=False)
        assert np.sum(attr) == pytest.approx(8., rel=self.limit, abs=0.)
        assert attr.shape == (2, )

        attr = self.pipeline.get_attribute('selected2', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(20, rel=self.limit, abs=0.)
        assert attr.shape == (2, 2)

        attr = self.pipeline.get_attribute('removed2', 'PARANG', static=False)
        assert np.sum(attr) == pytest.approx(16., rel=self.limit, abs=0.)
        assert attr.shape == (4, )

        attr = self.pipeline.get_attribute('removed2', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(40, rel=self.limit, abs=0.)
        assert attr.shape == (4, 2)

        module = FrameSelectionModule(name_in='select3',
                                      image_in_tag='start',
                                      selected_out_tag='selected3',
                                      removed_out_tag='removed3',
                                      index_out_tag='index3',
                                      method='range',
                                      threshold=(10., 10.7),
                                      fwhm=0.1,
                                      aperture=('circular', 0.1),
                                      position=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('select3')

        data = self.pipeline.get_data('selected3')
        assert np.sum(data) == pytest.approx(22.2568501695632, rel=self.limit, abs=0.)
        assert data.shape == (2, 11, 11)

        data = self.pipeline.get_data('removed3')
        assert np.sum(data) == pytest.approx(42.18622030593487, rel=self.limit, abs=0.)
        assert data.shape == (4, 11, 11)

        data = self.pipeline.get_data('index3')
        assert np.sum(data) == pytest.approx(12, rel=self.limit, abs=0.)
        assert data.shape == (4, )

    def test_image_statistics_full(self) -> None:

        module = ImageStatisticsModule(name_in='stat1',
                                       image_in_tag='read',
                                       stat_out_tag='stat1',
                                       position=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('stat1')

        data = self.pipeline.get_data('stat1')
        assert np.sum(data) == pytest.approx(115.68591492205017, rel=self.limit, abs=0.)
        assert data.shape == (10, 6)

    def test_image_statistics_posiiton(self) -> None:

        module = ImageStatisticsModule(name_in='stat2',
                                       image_in_tag='read',
                                       stat_out_tag='stat2',
                                       position=(5, 5, 0.1))

        self.pipeline.add_module(module)
        self.pipeline.run_module('stat2')

        data = self.pipeline.get_data('stat2')
        assert np.sum(data) == pytest.approx(118.7138708968444, rel=self.limit, abs=0.)
        assert data.shape == (10, 6)

    def test_frame_similarity_mse(self) -> None:

        module = FrameSimilarityModule(name_in='simi1',
                                       image_tag='read',
                                       method='MSE',
                                       mask_radius=(0., 0.2))

        self.pipeline.add_module(module)
        self.pipeline.run_module('simi1')

        attr = self.pipeline.get_attribute('read', 'MSE', static=False)
        assert np.min(attr) > 0.
        assert np.sum(attr) == pytest.approx(0.11739141370277852, rel=self.limit, abs=0.)
        assert attr.shape == (10, )

    def test_frame_similarity_pcc(self) -> None:

        module = FrameSimilarityModule(name_in='simi2',
                                       image_tag='read',
                                       method='PCC',
                                       mask_radius=(0., 0.2))

        self.pipeline.add_module(module)
        self.pipeline.run_module('simi2')

        attr = self.pipeline.get_attribute('read', 'PCC', static=False)
        assert np.min(attr) > 0.
        assert np.sum(attr) == pytest.approx(9.134820985662829, rel=self.limit, abs=0.)
        assert attr.shape == (10, )

    def test_frame_similarity_ssim(self) -> None:

        module = FrameSimilarityModule(name_in='simi3',
                                       image_tag='read',
                                       method='SSIM',
                                       mask_radius=(0., 0.2),
                                       temporal_median='constant')

        self.pipeline.add_module(module)
        self.pipeline.run_module('simi3')

        attr = self.pipeline.get_attribute('read', 'SSIM', static=False)
        assert np.min(attr) > 0.
        assert np.sum(attr) == pytest.approx(9.096830542868524, rel=self.limit, abs=0.)
        assert attr.shape == (10, )

    def test_select_by_attribute(self) -> None:

        self.pipeline.set_attribute('read', 'INDEX', np.arange(44), static=False)

        module = SelectByAttributeModule(name_in='frame_removal_1',
                                         image_in_tag='read',
                                         attribute_tag='SSIM',
                                         number_frames=6,
                                         order='descending',
                                         selected_out_tag='select_sim',
                                         removed_out_tag='remove_sim')

        self.pipeline.add_module(module)
        self.pipeline.run_module('frame_removal_1')

        attr = self.pipeline.get_attribute('select_sim', 'INDEX', static=False)
        assert np.sum(attr) == pytest.approx(946, rel=self.limit, abs=0.)
        assert attr.shape == (44, )

        attr = self.pipeline.get_attribute('select_sim', 'SSIM', static=False)
        assert np.sum(attr) == pytest.approx(5.556889532446573, rel=self.limit, abs=0.)
        assert attr.shape == (6, )

        attr = self.pipeline.get_attribute('remove_sim', 'SSIM', static=False)
        assert np.sum(attr) == pytest.approx(3.539941010421951, rel=self.limit, abs=0.)
        assert attr.shape == (4, )

    def test_residual_selection(self) -> None:

        module = ResidualSelectionModule(name_in='residual_select',
                                         image_in_tag='start',
                                         selected_out_tag='res_selected',
                                         removed_out_tag='res_removed',
                                         percentage=80.,
                                         annulus_radii=(0.1, 0.2))

        self.pipeline.add_module(module)
        self.pipeline.run_module('residual_select')

        data = self.pipeline.get_data('res_selected')
        assert np.sum(data) == pytest.approx(41.77295229983322, rel=self.limit, abs=0.)
        assert data.shape == (4, 11, 11)

        data = self.pipeline.get_data('res_removed')
        assert np.sum(data) == pytest.approx(22.670118175664847, rel=self.limit, abs=0.)
        assert data.shape == (2, 11, 11)
