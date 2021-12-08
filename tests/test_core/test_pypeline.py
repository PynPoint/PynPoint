import os

import pytest
import h5py
import numpy as np

from astropy.io import fits

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.readwrite.fitswriting import FitsWritingModule
from pynpoint.processing.badpixel import BadPixelSigmaFilterModule
from pynpoint.util.tests import create_config, remove_test_data


class TestPypeline:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        np.random.seed(1)
        images = np.random.normal(loc=0, scale=2e-4, size=(5, 11, 11))

        hdu = fits.PrimaryHDU()
        header = hdu.header
        header['INSTRUME'] = 'IMAGER'
        header['HIERARCH ESO DET EXP NO'] = 1
        header['HIERARCH ESO DET NDIT'] = 5
        header['HIERARCH ESO INS PIXSCALE'] = 0.01
        header['HIERARCH ESO ADA POSANG'] = 10.
        header['HIERARCH ESO ADA POSANG END'] = 20.
        header['HIERARCH ESO SEQ CUMOFFSETX'] = 5.
        header['HIERARCH ESO SEQ CUMOFFSETY'] = 5.
        hdu.data = images
        hdu.writeto(self.test_dir+'images.fits')

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, files=['images.fits'])

    def test_create_default_config(self) -> None:

        with pytest.warns(UserWarning) as warning:
            Pypeline(self.test_dir, self.test_dir, self.test_dir)

        assert len(warning) == 1

        assert warning[0].message.args[0] == 'Configuration file not found. Creating ' \
                                             'PynPoint_config.ini with default values ' \
                                             'in the working place.'

        with open(self.test_dir+'PynPoint_config.ini') as f_obj:
            count = 0
            for _ in f_obj:
                count += 1

        assert count == 25

    def test_create_none_config(self) -> None:

        file_obj = open(self.test_dir+'PynPoint_config.ini', 'w')
        file_obj.write('[header]\n\n')
        file_obj.write('INSTRUMENT: None\n')
        file_obj.write('NFRAMES: None\n')
        file_obj.write('EXP_NO: None\n')
        file_obj.write('NDIT: None\n')
        file_obj.write('PARANG_START: ESO ADA POSANG\n')
        file_obj.write('PARANG_END: None\n')
        file_obj.write('DITHER_X: None\n')
        file_obj.write('DITHER_Y: None\n')
        file_obj.write('DIT: None\n')
        file_obj.write('LATITUDE: None\n')
        file_obj.write('LONGITUDE: None\n')
        file_obj.write('PUPIL: None\n')
        file_obj.write('DATE: None\n')
        file_obj.write('RA: None\n')
        file_obj.write('DEC: None\n\n')
        file_obj.write('[settings]\n\n')
        file_obj.write('PIXSCALE: None\n')
        file_obj.write('MEMORY: None\n')
        file_obj.write('CPU: None\n')
        file_obj.close()

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        attribute = pipeline.get_attribute('config', 'MEMORY', static=True)
        assert attribute == 0

        attribute = pipeline.get_attribute('config', 'CPU', static=True)
        assert attribute == 0

        attribute = pipeline.get_attribute('config', 'PIXSCALE', static=True)
        assert attribute == pytest.approx(0., rel=self.limit, abs=0.)

        attribute = pipeline.get_attribute('config', 'INSTRUMENT', static=True)
        assert attribute == 'None'

        create_config(self.test_dir+'PynPoint_config.ini')

    def test_create_pipeline_path_missing(self) -> None:

        dir_non_exists = self.test_dir + 'none_dir/'
        dir_exists = self.test_dir

        with pytest.raises(AssertionError) as error:
            Pypeline(dir_non_exists, dir_exists, dir_exists)

        assert str(error.value) == 'The folder that was chosen for the working place does not ' \
                                   'exist: '+self.test_dir+'none_dir/.'

        with pytest.raises(AssertionError) as error:
            Pypeline(dir_exists, dir_non_exists, dir_exists)

        assert str(error.value) == 'The folder that was chosen for the input place does not ' \
                                   'exist: '+self.test_dir+'none_dir/.'

        with pytest.raises(AssertionError) as error:
            Pypeline(dir_exists, dir_exists, dir_non_exists)

        assert str(error.value) == 'The folder that was chosen for the output place does not ' \
                                   'exist: '+self.test_dir+'none_dir/.'

    def test_create_pipeline_existing_database(self) -> None:

        np.random.seed(1)
        images = np.random.normal(loc=0, scale=2e-4, size=(5, 11, 11))

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'w') as hdf_file:
            dset = hdf_file.create_dataset('images', data=images)
            dset.attrs['PIXSCALE'] = 0.01

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        data = pipeline.get_data('images')
        assert np.mean(data) == pytest.approx(1.1824138000882435e-05, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

        assert pipeline.get_attribute('images', 'PIXSCALE') == 0.01

        os.remove(self.test_dir+'PynPoint_database.hdf5')

    def test_create_pipeline_new_database(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        pipeline.m_data_storage.open_connection()
        pipeline.m_data_storage.close_connection()

        del pipeline

        os.remove(self.test_dir+'PynPoint_database.hdf5')

    def test_add_module(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        module = FitsReadingModule(name_in='read1',
                                   input_dir=None,
                                   image_tag='im_arr1')

        assert pipeline.add_module(module) is None

        module = FitsReadingModule(name_in='read2',
                                   input_dir=self.test_dir,
                                   image_tag='im_arr2')

        assert pipeline.add_module(module) is None

        with pytest.warns(UserWarning) as warning:
            pipeline.add_module(module)

        assert len(warning) == 1

        assert warning[0].message.args[0] == 'Names of pipeline modules that are added to the ' \
                                             'Pypeline need to be unique. The current pipeline ' \
                                             'module, \'read2\', does already exist in the ' \
                                             'Pypeline dictionary so the previous module with ' \
                                             'the same name will be overwritten.'

        module = BadPixelSigmaFilterModule(name_in='badpixel',
                                           image_in_tag='im_arr1',
                                           image_out_tag='im_out')

        assert pipeline.add_module(module) is None

        module = FitsWritingModule(name_in='write1',
                                   file_name='result.fits',
                                   data_tag='im_arr1')

        assert pipeline.add_module(module) is None

        module = FitsWritingModule(name_in='write2',
                                   file_name='result.fits',
                                   data_tag='im_arr1',
                                   output_dir=self.test_dir)

        assert pipeline.add_module(module) is None

        assert pipeline.run() is None

        assert pipeline.get_module_names() == ['read1', 'read2', 'badpixel', 'write1', 'write2']

        os.remove(self.test_dir+'result.fits')
        os.remove(self.test_dir+'PynPoint_database.hdf5')

    def test_run_module(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        module = FitsReadingModule(name_in='read',
                                   image_tag='im_arr')

        assert pipeline.add_module(module) is None
        assert pipeline.run_module('read') is None

        os.remove(self.test_dir+'PynPoint_database.hdf5')

    def test_add_wrong_module(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        with pytest.raises(TypeError) as error:
            pipeline.add_module(None)

        assert str(error.value) == 'type of argument "module" must be ' \
                                   'pynpoint.core.processing.PypelineModule; got NoneType instead'

        os.remove(self.test_dir+'PynPoint_database.hdf5')

    def test_run_module_wrong_tag(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        module = FitsReadingModule(name_in='read')

        pipeline.add_module(module)

        module = FitsWritingModule(name_in='write',
                                   file_name='result.fits',
                                   data_tag='im_list')

        pipeline.add_module(module)

        module = BadPixelSigmaFilterModule(name_in='badpixel',
                                           image_in_tag='im_list',
                                           image_out_tag='im_out')

        pipeline.add_module(module)

        with pytest.raises(AttributeError) as error:
            pipeline.run_module('badpixel')

        assert str(error.value) == 'Pipeline module \'badpixel\' is looking for data under a ' \
                                   'tag which does not exist in the database.'

        with pytest.raises(AttributeError) as error:
            pipeline.run_module('write')

        assert str(error.value) == 'Pipeline module \'write\' is looking for data under a tag ' \
                                   'which does not exist in the database.'

        with pytest.raises(AttributeError) as error:
            pipeline.run()

        assert str(error.value) == 'Pipeline module \'write\' is looking for data under a tag ' \
                                   'which is not created by a previous module or the data does ' \
                                   'not exist in the database.'

        assert pipeline.validate_pipeline_module('test') == (False, 'test')

        with pytest.raises(TypeError) as error:
            pipeline._validate('module', 'tag')

        assert str(error.value) == 'type of argument "module" must be one of (ReadingModule, ' \
                                   'WritingModule, ProcessingModule); got str instead'

        os.remove(self.test_dir+'PynPoint_database.hdf5')

    def test_run_module_non_existing(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        with pytest.warns(UserWarning) as warning:
            pipeline.run_module('test')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'Pipeline module \'test\' not found.'

        os.remove(self.test_dir+'PynPoint_database.hdf5')

    def test_remove_module(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        module = FitsReadingModule(name_in='read')

        pipeline.add_module(module)

        module = BadPixelSigmaFilterModule(name_in='badpixel',
                                           image_in_tag='im_arr1',
                                           image_out_tag='im_out')

        pipeline.add_module(module)

        assert pipeline.get_module_names() == ['read', 'badpixel']
        assert pipeline.remove_module('read')

        assert pipeline.get_module_names() == ['badpixel']
        assert pipeline.remove_module('badpixel')

        with pytest.warns(UserWarning) as warning:
            pipeline.remove_module('test')

        assert len(warning) == 1

        assert warning[0].message.args[0] == 'Pipeline module \'test\' is not found in the ' \
                                             'Pypeline dictionary so it could not be removed. ' \
                                             'The dictionary contains the following modules: [].' \

        os.remove(self.test_dir+'PynPoint_database.hdf5')

    def test_get_shape(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        module = FitsReadingModule(name_in='read',
                                   image_tag='images')

        pipeline.add_module(module)
        pipeline.run_module('read')

        assert pipeline.get_shape('images') == (5, 11, 11)

    def test_get_tags(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        assert pipeline.get_tags() == ['images']

    def test_list_attributes(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        attr_dict = pipeline.list_attributes('images')

        assert len(attr_dict) == 11
        assert attr_dict['INSTRUMENT'] == 'IMAGER'
        assert attr_dict['PIXSCALE'] == 0.027
        assert attr_dict['NFRAMES'] == [5]
        assert attr_dict['PARANG_START'] == [10.]

    def test_set_and_get_attribute(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        pipeline.set_attribute('images', 'PIXSCALE', 0.1, static=True)
        pipeline.set_attribute('images', 'PARANG', np.arange(1., 11., 1.), static=False)

        attribute = pipeline.get_attribute('images', 'PIXSCALE', static=True)
        assert attribute == pytest.approx(0.1, rel=self.limit, abs=0.)

        attribute = pipeline.get_attribute('images', 'PARANG', static=False)
        assert attribute == pytest.approx(np.arange(1., 11., 1.), rel=self.limit, abs=0.)

        pipeline.set_attribute('images', 'PARANG', np.arange(10., 21., 1.), static=False)

        attribute = pipeline.get_attribute('images', 'PARANG', static=False)
        assert attribute == pytest.approx(np.arange(10., 21., 1.), rel=self.limit, abs=0.)

    def test_get_data_range(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        data = pipeline.get_data('images', data_range=(0, 2))

        assert data.shape == (2, 11, 11)

    def test_delete_data(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        pipeline.delete_data('images')

        assert len(pipeline.get_tags()) == 0

    def test_delete_not_found(self) -> None:

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        with pytest.warns(UserWarning) as warning:
            pipeline.delete_data('images')

        assert len(warning) == 2

        assert warning[0].message.args[0] == 'Dataset \'images\' not found in the database.'
        assert warning[1].message.args[0] == 'Attributes of \'images\' not found in the database.'

    def test_omp_num_threads(self) -> None:

        os.environ['OMP_NUM_THREADS'] = '2'

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)
        pipeline.run()
