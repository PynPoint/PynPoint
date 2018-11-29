"""
Functions for the test cases.
"""

from __future__ import absolute_import

import os
import math
import shutil

import h5py
import numpy as np

from scipy.ndimage import shift
from astropy.io import fits
from six.moves import range


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
    file_obj.write('DITHER_X: ESO SEQ CUMOFFSETX\n')
    file_obj.write('DITHER_Y: ESO SEQ CUMOFFSETY\n')
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
    file_obj.write('CPU: 1\n')

    file_obj.close()

def create_random(path,
                  ndit=10,
                  parang=np.arange(1., 11., 1.)):
    """
    Create a stack of images with Gaussian distributed pixel values.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    file_in = path + "/PynPoint_database.hdf5"

    np.random.seed(1)
    images = np.random.normal(loc=0, scale=2e-4, size=(ndit, 100, 100))

    h5f = h5py.File(file_in, "w")
    dset = h5f.create_dataset("images", data=images)
    dset.attrs['PIXSCALE'] = 0.01
    if parang is not None:
        h5f.create_dataset("header_images/PARANG", data=parang)
    h5f.close()

def create_fits(path,
                filename,
                image,
                ndit,
                exp_no=0,
                parang=[0., 0.],
                x0=0.,
                y0=0.):
    """
    Create a FITS file with images and header information.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = 'IMAGER'
    header['HIERARCH ESO DET EXP NO'] = 1.
    header['HIERARCH ESO DET NDIT'] = ndit
    header['HIERARCH ESO DET EXP NO'] = exp_no
    header['HIERARCH ESO ADA POSANG'] = parang[0]
    header['HIERARCH ESO ADA POSANG END'] = parang[1]
    header['HIERARCH ESO SEQ CUMOFFSETX'] = x0
    header['HIERARCH ESO SEQ CUMOFFSETY'] = y0
    hdu.data = image
    hdu.writeto(os.path.join(path, filename))

def create_fake(path,
                ndit,
                nframes,
                exp_no,
                npix,
                fwhm,
                x0,
                y0,
                angles,
                sep,
                contrast):
    """
    Create ADI test data with fake planets.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    parang = []
    for i, item in enumerate(angles):
        for j in range(ndit[i]):
            parang.append(item[0]+float(j)*(item[1]-item[0])/float(ndit[i]))

    if fwhm is not None or contrast is not None:
        sigma = fwhm / (2.*math.sqrt(2.*math.log(2.)))

    x = np.arange(0., npix[0], 1.)
    y = np.arange(0., npix[1], 1.)
    xx, yy = np.meshgrid(x, y)

    np.random.seed(1)

    count = 0
    for j, item in enumerate(nframes):
        image = np.zeros((item, npix[1], npix[0]))

        for i in range(ndit[j]):
            noise = np.random.normal(loc=0, scale=2e-4, size=(npix[1], npix[0]))
            image[i, 0:npix[1], 0:npix[0]] = noise

            if fwhm is not None:
                star = (1./(2.*np.pi*sigma**2))*np.exp(-((xx-x0[j])**2+(yy-y0[j])**2)/(2.*sigma**2))
                image[i, 0:npix[1], 0:npix[0]] += star

            if contrast is not None and sep is not None:
                planet = contrast*(1./(2.*np.pi*sigma**2))*np.exp(-((xx-x0[j])**2+(yy-y0[j])**2)/ \
                         (2.*sigma**2))
                x_shift = sep*math.cos(parang[count]*math.pi/180.)
                y_shift = sep*math.sin(parang[count]*math.pi/180.)
                planet = shift(planet, (x_shift, y_shift), order=5)
                image[i, 0:npix[1], 0:npix[0]] += planet

            count += 1

        create_fits(path, 'image'+str(j+1).zfill(2)+'.fits', image, ndit[j], exp_no[j],
                    angles[j], x0[j]-npix[0]/2., y0[j]-npix[1]/2.)

def create_star_data(path,
                     npix_x=100,
                     npix_y=100,
                     x0=[50., 50., 50., 50.],
                     y0=[50., 50., 50., 50.],
                     parang_start=[0., 5., 10., 15.],
                     parang_end=[5., 10., 15., 20.],
                     exp_no=[1, 2, 3, 4],
                     ndit=10,
                     nframes=10,
                     noise=True):
    """
    Create data with a stellar PSF and Gaussian noise.
    """

    fwhm = 3.

    if not os.path.exists(path):
        os.makedirs(path)

    np.random.seed(1)

    for j, item in enumerate(exp_no):
        sigma = fwhm / (2. * math.sqrt(2.*math.log(2.)))

        x = y = np.arange(0., npix_x, 1.)
        xx, yy = np.meshgrid(x, y)

        image = np.zeros((nframes, npix_x, npix_y))

        for i in range(nframes):
            image[i, ] = (1./(2.*np.pi*sigma**2))*np.exp(-((xx-x0[j])**2+(yy-y0[j])**2)/ \
                         (2.*sigma**2))
            if noise:
                image[i, ] += np.random.normal(loc=0, scale=2e-4, size=(npix_x, npix_x))

        hdu = fits.PrimaryHDU()
        header = hdu.header
        header['INSTRUME'] = 'IMAGER'
        header['HIERARCH ESO DET EXP NO'] = item
        header['HIERARCH ESO DET NDIT'] = ndit
        header['HIERARCH ESO ADA POSANG'] = parang_start[j]
        header['HIERARCH ESO ADA POSANG END'] = parang_end[j]
        header['HIERARCH ESO SEQ CUMOFFSETX'] = "None"
        header['HIERARCH ESO SEQ CUMOFFSETY'] = "None"
        hdu.data = image
        hdu.writeto(os.path.join(path, 'image'+str(j+1).zfill(2)+'.fits'))

def create_waffle_data(path,
                       npix,
                       x_waffle,
                       y_waffle):
    """
    Create data with waffle spots and Gaussian noise.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    fwhm = 3

    sigma = fwhm / (2. * math.sqrt(2.*math.log(2.)))

    x = y = np.arange(0., npix, 1.)
    xx, yy = np.meshgrid(x, y)

    image = np.zeros((npix, npix))

    for j, _ in enumerate(x_waffle):
        star = (1./(2.*np.pi*sigma**2))*np.exp(-((xx-x_waffle[j])**2+(yy-y_waffle[j])**2)/ \
               (2.*sigma**2))
        image += star

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = 'IMAGER'
    header['HIERARCH ESO DET EXP NO'] = "None"
    header['HIERARCH ESO DET NDIT'] = "none"
    header['HIERARCH ESO ADA POSANG'] = "None"
    header['HIERARCH ESO ADA POSANG END'] = "None"
    header['HIERARCH ESO SEQ CUMOFFSETX'] = "None"
    header['HIERARCH ESO SEQ CUMOFFSETY'] = "None"
    hdu.data = image
    hdu.writeto(os.path.join(path, 'image01.fits'))

def remove_test_data(path,
                     folders=None,
                     files=None):
    """
    Function to remove data created by the test cases.
    """

    os.remove(path+'PynPoint_database.hdf5')
    os.remove(path+'PynPoint_config.ini')

    if folders is not None:
        for item in folders:
            shutil.rmtree(path+item)

    if files is not None:
        for item in files:
            os.remove(path+item)
            