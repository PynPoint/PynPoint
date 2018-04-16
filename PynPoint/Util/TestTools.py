"""
Functions for the test cases.
"""

import numpy as np
from astropy.io import fits


def create_config(filename):
    """
    Create a configuration file.
    """

    file_obj = open(filename, 'w')

    file_obj.write('[header]\n\n')
    file_obj.write('INSTRUMENT: INSTRUME\n')
    file_obj.write('NFRAMES: NAXIS3\n')
    file_obj.write('EXP_NO: ESO DET EXP NO\n')
    file_obj.write('NDIT: ESO DET NDIT\n')
    file_obj.write('PARANG_START: ESO ADA POSANG\n')
    file_obj.write('PARANG_END: ESO ADA POSANG END\n')
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
    file_obj.write('PIXSCALE: 0.027\n')
    file_obj.write('MEMORY: 100\n')
    file_obj.write('CPU: 1')

    file_obj.close()

def prepare_pca_tests(path):
    """
    Create the images and configuration file for the test cases of the PCA PSF subtraction.
    """

    image1 = np.loadtxt(path + "/test_data/image1.dat")
    image2 = np.loadtxt(path + "/test_data/image2.dat")
    image3 = np.loadtxt(path + "/test_data/image3.dat")
    image4 = np.loadtxt(path + "/test_data/image4.dat")

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = "IMAGER"
    header['HIERARCH ESO DET EXP NO'] = 1
    header['HIERARCH ESO DET NDIT'] = 1
    header['HIERARCH ESO ADA POSANG'] = 1.
    header['HIERARCH ESO ADA POSANG END'] = 1.
    header['PARANG'] = -17.3261
    header['PARANG'] = -17.3261
    hdu.data = image1
    hdu.writeto(path+"/test_data/image1.fits")

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = "IMAGER"
    header['HIERARCH ESO DET EXP NO'] = 2
    header['HIERARCH ESO DET NDIT'] = 1
    header['HIERARCH ESO ADA POSANG'] = 1.
    header['HIERARCH ESO ADA POSANG END'] = 1.
    header['PARANG'] = -17.1720
    hdu.data = image2
    hdu.writeto(path+"/test_data/image2.fits")

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = "IMAGER"
    header['HIERARCH ESO DET EXP NO'] = 3
    header['HIERARCH ESO DET NDIT'] = 1
    header['HIERARCH ESO ADA POSANG'] = 1.
    header['HIERARCH ESO ADA POSANG END'] = 1.
    header['PARANG'] = -17.0143
    hdu.data = image3
    hdu.writeto(path+"/test_data/image3.fits")

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = "IMAGER"
    header['HIERARCH ESO DET EXP NO'] = 4
    header['HIERARCH ESO DET NDIT'] = 1
    header['HIERARCH ESO ADA POSANG'] = 1.
    header['HIERARCH ESO ADA POSANG END'] = 1.
    header['PARANG'] = -16.6004
    hdu.data = image4
    hdu.writeto(path+"/test_data/image4.fits")

    config_file = path+"/test_data/PynPoint_config.ini"

    file_obj = open(config_file, 'w')
    file_obj.write('[header]\n\n')
    file_obj.write('INSTRUMENT: INSTRUME\n')
    file_obj.write('NFRAMES: NAXIS3\n')
    file_obj.write('EXP_NO: ESO DET EXP NO\n')
    file_obj.write('NDIT: ESO DET NDIT\n')
    file_obj.write('PARANG_START: ESO ADA POSANG\n')
    file_obj.write('PARANG_END: ESO ADA POSANG END\n')
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
    file_obj.write('PIXSCALE: 0.01\n')
    file_obj.write('MEMORY: 100\n')
    file_obj.write('CPU: 1')
    file_obj.close()
