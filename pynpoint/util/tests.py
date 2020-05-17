"""
Functions for testing the pipeline and its modules.
"""

import os
import math
import shutil
import subprocess

from typing import List, Optional, Tuple, Union

import h5py
import numpy as np

from typeguard import typechecked
from astropy.io import fits
from scipy.ndimage import shift


@typechecked
def create_config(filename: str) -> None:
    """
    Create a configuration file.

    Parameters
    ----------
    filename : str
        Configuration filename.

    Returns
    -------
    NoneType
        None
    """

    with open(filename, 'w') as file_obj:

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
        file_obj.write('MEMORY: 39\n')
        file_obj.write('CPU: 1\n')


@typechecked
def create_random(path: str,
                  nimages: float = 5) -> None:
    """
    Create a dataset of images with Gaussian distributed pixel values.

    Parameters
    ----------
    path : str
        Working folder.
    nimages : int
        Number of images.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(path):
        os.makedirs(path)

    file_in = os.path.join(path, 'PynPoint_database.hdf5')

    np.random.seed(1)
    images = np.random.normal(loc=0, scale=2e-4, size=(nimages, 11, 11))

    with h5py.File(file_in, 'w') as h5_file:
        dset = h5_file.create_dataset('images', data=images)
        dset.attrs['PIXSCALE'] = 0.01

        h5_file.create_dataset('header_images/PARANG', data=np.arange(float(nimages)))


@typechecked
def create_fits(path: str,
                filename: str,
                image: np.ndarray,
                ndit: int,
                exp_no: int,
                parang: np.ndarray,
                x0: float,
                y0: float) -> None:
    """
    Create a FITS file with images and header information.

    Parameters
    ----------
    path : str
        Working folder.
    filename : str
        FITS filename.
    image : np.ndarray
        Images.
    ndit : int
        Number of integrations.
    exp_no : int
        Exposure number.
    parang : list(float, float)
        Start and end parallactic angle.
    x0 : float
        Horizontal dither position.
    y0 : float
        Vertical dither position.

    Returns
    -------
    NoneType
        None
    """

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


@typechecked
def create_fake_data(path: str) -> None:
    """
    Create ADI test data with a fake planet.

    Parameters
    ----------
    path : str
        Working folder.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(path):
        os.makedirs(path)

    ndit = 10
    npix = 21
    fwhm = 3.
    sep = 6.
    contrast = 1e-2
    pos_star = 10.
    exp_no = 1
    parang = np.linspace(0., 180., 10)

    np.random.seed(1)

    sigma = fwhm / (2.*math.sqrt(2.*math.log(2.)))

    x = np.arange(0., 21., 1.)
    y = np.arange(0., 21., 1.)

    xx, yy = np.meshgrid(x, y)

    images = np.zeros((ndit, npix, npix))

    for i, item in enumerate(parang):
        images[i, ] = np.random.normal(loc=0, scale=2e-4, size=(npix, npix))

        star = np.exp(-((xx-pos_star)**2+(yy-pos_star)**2)/(2.*sigma**2))/(2.*np.pi*sigma**2)

        x_shift = sep*math.cos(math.radians(item))
        y_shift = sep*math.sin(math.radians(item))

        images[i, ] += star + shift(contrast*star, (x_shift, y_shift), order=5)

    create_fits(path, 'images.fits', images, ndit, exp_no, parang, pos_star, pos_star)


@typechecked
def create_star_data(path: str,
                     npix: int = 11,
                     pos_star: float = 5.) -> None:
    """
    Create a dataset with a PSF and Gaussian noise.

    Parameters
    ----------
    path : str
        Working folder.
    npix : int
        Number of pixels in each dimension.

    Returns
    -------
    NoneType
        None
    """

    fwhm = 3.
    nimages = 5

    exp_no = [1, 2]
    parang_start = [0., 90.]
    parang_end = [90., 180.]

    if not os.path.exists(path):
        os.makedirs(path)

    np.random.seed(1)

    for j, exp_item in enumerate(exp_no):
        sigma = fwhm / (2. * math.sqrt(2.*math.log(2.)))

        x = y = np.arange(0., float(npix), 1.)
        xx, yy = np.meshgrid(x, y)

        image = np.zeros((nimages, npix, npix))

        for i in range(nimages):
            image[i, ] = np.exp(-((xx-pos_star)**2+(yy-pos_star)**2)/(2.*sigma**2)) / (2.*np.pi*sigma**2)
            image[i, ] += np.random.normal(loc=0, scale=2e-4, size=(npix, npix))

        hdu = fits.PrimaryHDU()
        header = hdu.header
        header['INSTRUME'] = 'IMAGER'
        header['HIERARCH ESO DET EXP NO'] = exp_item
        header['HIERARCH ESO DET NDIT'] = nimages
        header['HIERARCH ESO ADA POSANG'] = parang_start[j]
        header['HIERARCH ESO ADA POSANG END'] = parang_end[j]
        header['HIERARCH ESO SEQ CUMOFFSETX'] = 'None'
        header['HIERARCH ESO SEQ CUMOFFSETY'] = 'None'
        hdu.data = image
        hdu.writeto(os.path.join(path, f'images_{j}.fits'))


@typechecked
def create_waffle_data(path: str,
                       npix: int,
                       x_spot: List[float],
                       y_spot: List[float]) -> None:
    """
    Create data with satellite spots and Gaussian noise.

    Parameters
    ----------
    path : str
        Working folder.
    npix : int
        Number of pixels in both dimensions.
    x_spot : list(float, )
        Pixel positions in horizontal direction of the satellite spots.
    y_spot : list(float, )
        Pixel positions in vertical direction of the satellite spots.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(path):
        os.makedirs(path)

    fwhm = 3

    sigma = fwhm / (2. * math.sqrt(2.*math.log(2.)))

    x = y = np.arange(0., npix, 1.)
    xx, yy = np.meshgrid(x, y)

    image = np.zeros((npix, npix))

    for j, _ in enumerate(x_spot):
        star = (1./(2.*np.pi*sigma**2))*np.exp(-((xx-x_spot[j])**2+(yy-y_spot[j])**2) /
                                               (2.*sigma**2))
        image += star

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = 'IMAGER'
    header['HIERARCH ESO DET EXP NO'] = 'None'
    header['HIERARCH ESO DET NDIT'] = 'none'
    header['HIERARCH ESO ADA POSANG'] = 'None'
    header['HIERARCH ESO ADA POSANG END'] = 'None'
    header['HIERARCH ESO SEQ CUMOFFSETX'] = 'None'
    header['HIERARCH ESO SEQ CUMOFFSETY'] = 'None'
    hdu.data = image
    hdu.writeto(os.path.join(path, 'images.fits'))


@typechecked
def remove_test_data(path: str,
                     folders: Optional[List[str]] = None,
                     files: Optional[List[str]] = None) -> None:
    """
    Function to remove data created by the test cases.

    Parameters
    ----------
    path : str
        Working folder.
    folders : list(str, )
        Folders to remove.
    files : list(str, )
        Files to removes.

    Returns
    -------
    NoneType
        None
    """

    os.remove(path+'PynPoint_database.hdf5')
    os.remove(path+'PynPoint_config.ini')

    if folders is not None:
        for item in folders:
            shutil.rmtree(path+item)

    if files is not None:
        for item in files:
            os.remove(path+item)


@typechecked
def create_near_data(path: str) -> None:
    """
    Create a stack of images with Gaussian distributed pixel values.

    Parameters
    ----------
    path : str
        Working folder.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(path):
        os.makedirs(path)

    np.random.seed(1)

    image = np.random.normal(loc=0., scale=1., size=(10, 10))

    exp_no = [1, 2]

    for i, item in enumerate(exp_no):
        fits_file = os.path.join(path, f'images_{i}.fits')

        primary_header = fits.Header()
        primary_header['INSTRUME'] = 'VISIR'
        primary_header['HIERARCH ESO DET CHOP NCYCLES'] = 5
        primary_header['HIERARCH ESO DET SEQ1 DIT'] = 1.
        primary_header['HIERARCH ESO TPL EXPNO'] = item
        primary_header['HIERARCH ESO DET CHOP ST'] = 'T'
        primary_header['HIERARCH ESO DET CHOP CYCSKIP'] = 0
        primary_header['HIERARCH ESO DET CHOP CYCSUM'] = 'F'

        chopa_header = fits.Header()
        chopa_header['HIERARCH ESO DET FRAM TYPE'] = 'HCYCLE1'

        chopb_header = fits.Header()
        chopb_header['HIERARCH ESO DET FRAM TYPE'] = 'HCYCLE2'

        hdu = [fits.PrimaryHDU(header=primary_header)]

        for _ in range(5):
            hdu.append(fits.ImageHDU(image, header=chopa_header))
            hdu.append(fits.ImageHDU(image, header=chopb_header))

        # last image is the average of all images
        hdu.append(fits.ImageHDU(image))

        hdulist = fits.HDUList(hdu)
        hdulist.writeto(fits_file)

        subprocess.call('compress '+fits_file, shell=True)
