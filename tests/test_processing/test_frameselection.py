import os
import warnings

import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.frameselection import RemoveFramesModule, FrameSelectionModule, \
                                               RemoveLastFrameModule, RemoveStartFramesModule
from pynpoint.util.tests import create_config, remove_test_data, create_star_data

warnings.simplefilter("always")

limit = 1e-10

class TestFrameSelection(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"images", ndit=10, nframes=11)
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["images"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read",
                                 input_dir=self.test_dir+"images",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)
        self.pipeline.run_module("read")

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0001002167910262529, rtol=limit, atol=0.)
        assert data.shape == (44, 100, 100)

    def test_remove_last_frame(self):

        last = RemoveLastFrameModule(name_in="last",
                                     image_in_tag="read",
                                     image_out_tag="last")

        self.pipeline.add_module(last)
        self.pipeline.run_module("last")

        data = self.pipeline.get_data("last")
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010020258903646778, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        self.pipeline.set_attribute("last", "PARANG", np.arange(0., 40., 1.), static=False)

        star = np.zeros((40, 2))
        star[:, 0] = np.arange(40., 80., 1.)
        star[:, 1] = np.arange(40., 80., 1.)

        self.pipeline.set_attribute("last", "STAR_POSITION", star, static=False)

        attribute = self.pipeline.get_attribute("last", "PARANG", static=False)
        assert np.allclose(np.mean(attribute), 19.5, rtol=limit, atol=0.)
        assert attribute.shape == (40, )

        attribute = self.pipeline.get_attribute("last", "STAR_POSITION", static=False)
        assert np.allclose(np.mean(attribute), 59.5, rtol=limit, atol=0.)
        assert attribute.shape == (40, 2)

    def test_remove_start_frame(self):

        start = RemoveStartFramesModule(frames=2,
                                        name_in="start",
                                        image_in_tag="last",
                                        image_out_tag="start")

        self.pipeline.add_module(start)
        self.pipeline.run_module("start")

        data = self.pipeline.get_data("start")
        assert np.allclose(data[0, 50, 50], 0.09797376304048713, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010011298467340513, rtol=limit, atol=0.)
        assert data.shape == (32, 100, 100)

        attribute = self.pipeline.get_attribute("start", "PARANG", static=False)
        assert np.allclose(np.mean(attribute), 20.5, rtol=limit, atol=0.)
        assert attribute.shape == (32, )

        attribute = self.pipeline.get_attribute("start", "STAR_POSITION", static=False)
        assert np.allclose(np.mean(attribute), 60.5, rtol=limit, atol=0.)
        assert attribute.shape == (32, 2)

    def test_remove_frames(self):

        remove = RemoveFramesModule(frames=(5, 8, 13, 25, 31),
                                    name_in="remove",
                                    image_in_tag="start",
                                    selected_out_tag="selected",
                                    removed_out_tag="removed")

        self.pipeline.add_module(remove)
        self.pipeline.run_module("remove")

        data = self.pipeline.get_data("selected")
        assert np.allclose(data[0, 50, 50], 0.09797376304048713, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.984682304434105e-05, rtol=limit, atol=0.)
        assert data.shape == (27, 100, 100)

        data = self.pipeline.get_data("removed")
        assert np.allclose(data[0, 50, 50], 0.09818692015286978, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010155025747035087, rtol=limit, atol=0.)
        assert data.shape == (5, 100, 100)

        attribute = self.pipeline.get_attribute("selected", "PARANG", static=False)
        assert np.allclose(np.mean(attribute), 20.296296296296298, rtol=limit, atol=0.)
        assert attribute.shape == (27, )

        attribute = self.pipeline.get_attribute("selected", "STAR_POSITION", static=False)
        assert np.allclose(np.mean(attribute), 60.2962962962963, rtol=limit, atol=0.)
        assert attribute.shape == (27, 2)

        attribute = self.pipeline.get_attribute("removed", "PARANG", static=False)
        assert np.allclose(np.mean(attribute), 21.6, rtol=limit, atol=0.)
        assert attribute.shape == (5, )

        attribute = self.pipeline.get_attribute("removed", "STAR_POSITION", static=False)
        assert np.allclose(np.mean(attribute), 61.6, rtol=limit, atol=0.)
        assert attribute.shape == (5, 2)

    def test_frame_selection(self):

        select = FrameSelectionModule(name_in="select1",
                                      image_in_tag="start",
                                      selected_out_tag="selected1",
                                      removed_out_tag="removed1",
                                      index_out_tag="index1",
                                      method="median",
                                      threshold=1.,
                                      fwhm=0.1,
                                      aperture=("circular", 0.2),
                                      position=(None, None, 0.5))

        self.pipeline.add_module(select)
        self.pipeline.run_module("select1")

        data = self.pipeline.get_data("selected1")
        assert np.allclose(data[0, 50, 50], 0.09791350617182591, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.980792188317311e-05, rtol=limit, atol=0.)
        assert data.shape == (22, 100, 100)

        data = self.pipeline.get_data("removed1")
        assert np.allclose(data[0, 50, 50], 0.09797376304048713, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010078412281191547, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

        data = self.pipeline.get_data("index1")
        assert data[-1] == 28
        assert np.sum(data) == 115
        assert data.shape == (10, )

        attribute = self.pipeline.get_attribute("selected1", "PARANG", static=False)
        assert np.allclose(np.mean(attribute), 22.681818181818183, rtol=limit, atol=0.)
        assert attribute.shape == (22, )

        attribute = self.pipeline.get_attribute("selected1", "STAR_POSITION", static=False)
        assert np.allclose(np.mean(attribute), 50.0, rtol=limit, atol=0.)
        assert attribute.shape == (22, 2)

        attribute = self.pipeline.get_attribute("removed1", "PARANG", static=False)
        assert np.allclose(np.mean(attribute), 15.7, rtol=limit, atol=0.)
        assert attribute.shape == (10, )

        attribute = self.pipeline.get_attribute("removed1", "STAR_POSITION", static=False)
        assert np.allclose(np.mean(attribute), 50.0, rtol=limit, atol=0.)
        assert attribute.shape == (10, 2)

        select = FrameSelectionModule(name_in="select2",
                                      image_in_tag="start",
                                      selected_out_tag="selected2",
                                      removed_out_tag="removed2",
                                      index_out_tag="index2",
                                      method="max",
                                      threshold=3.,
                                      fwhm=0.1,
                                      aperture=("annulus", 0.1, 0.2),
                                      position=None)

        self.pipeline.add_module(select)
        self.pipeline.run_module("select2")

        data = self.pipeline.get_data("selected2")
        assert np.allclose(data[0, 50, 50], 0.09797376304048713, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010037996502199598, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("removed2")
        assert np.allclose(data[0, 50, 50], 0.097912284606689, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.966801742575358e-05, rtol=limit, atol=0.)
        assert data.shape == (12, 100, 100)

        data = self.pipeline.get_data("index2")
        assert data[-1] == 30
        assert np.sum(data) == 230
        assert data.shape == (12, )

        attribute = self.pipeline.get_attribute("selected2", "PARANG", static=False)
        assert np.allclose(np.mean(attribute), 17.8, rtol=limit, atol=0.)
        assert attribute.shape == (20, )

        attribute = self.pipeline.get_attribute("selected2", "STAR_POSITION", static=False)
        assert np.allclose(np.mean(attribute), 50.0, rtol=limit, atol=0.)
        assert attribute.shape == (20, 2)

        attribute = self.pipeline.get_attribute("removed2", "PARANG", static=False)
        assert np.allclose(np.mean(attribute), 25.0, rtol=limit, atol=0.)
        assert attribute.shape == (12, )

        attribute = self.pipeline.get_attribute("removed2", "STAR_POSITION", static=False)
        assert np.allclose(np.mean(attribute), 50.0, rtol=limit, atol=0.)
        assert attribute.shape == (12, 2)
