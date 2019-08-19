"""
Functions for testing the pipeline and modules.
"""

import os
import math
import shutil
import subprocess

from typing import Union, List, Tuple

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
                  ndit: int = 10,
                  parang: Union[np.ndarray, None] = np.arange(1., 11., 1.)) -> None:
    """
    Create a stack of images with Gaussian distributed pixel values.

    Parameters
    ----------
    path : str
        Working folder.
    ndit : int
        Number of images.
    parang : numpy.ndarray, None
        Parallactic angles.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(path):
        os.makedirs(path)

    file_in = path + '/PynPoint_database.hdf5'

    np.random.seed(1)
    images = np.random.normal(loc=0, scale=2e-4, size=(ndit, 100, 100))

    h5f = h5py.File(file_in, 'w')
    dset = h5f.create_dataset('images', data=images)
    dset.attrs['PIXSCALE'] = 0.01
    if parang is not None:
        h5f.create_dataset('header_images/PARANG', data=parang)
    h5f.close()


@typechecked
def create_fits(path: str,
                filename: str,
                image: np.ndarray,
                ndit: int,
                exp_no: int = 0,
                parang: List[float] = [0., 0.],
                x0: float = 0.,
                y0: float = 0.) -> None:
    """
    Create a FITS file with images and header information.

    Parameters
    ----------
    path : str
        Working folder.
    filename : str
        FITS filename.
    image : numpy.ndarray
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
def create_fake(path: str,
                ndit: List[int],
                nframes: List[int],
                exp_no: List[int],
                npix: Tuple[int, int],
                fwhm: Union[float, None],
                x0: List[float],
                y0: List[float],
                angles: List[List[float]],
                sep: Union[float, None],
                contrast: Union[float, None]) -> None:
    """
    Create ADI test data with a fake planet.

    Parameters
    ----------
    path : str
        Working folder.
    ndit : list(int, )
        Number of exposures.
    nframes : list(int, )
        Number of images.
    exp_no : list(int, )
        Exposure numbers.
    npix : tupe(int, int)
        Number of pixels in x and y direction.
    fwhm : float, None
        Full width at half maximum (pix) of the PSF.
    x0 : list(float, )
        Horizontal positions of the star.
    y0 : list(float, )
        Vertical positions of the star.
    angles : list(list(float, float), )
        Derotation angles (deg).
    sep : float, None
        Separation of the planet.
    contrast : float, None
        Brightness contrast of the planet.

    Returns
    -------
    NoneType
        None
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
                planet = contrast*(1./(2.*np.pi*sigma**2))*np.exp(-((xx-x0[j])**2+(yy-y0[j])**2) /
                                                                  (2.*sigma**2))
                x_shift = sep*math.cos(parang[count]*math.pi/180.)
                y_shift = sep*math.sin(parang[count]*math.pi/180.)
                planet = shift(planet, (x_shift, y_shift), order=5)
                image[i, 0:npix[1], 0:npix[0]] += planet

            count += 1

        create_fits(path, 'image'+str(j+1).zfill(2)+'.fits', image, ndit[j], exp_no[j],
                    angles[j], x0[j]-npix[0]/2., y0[j]-npix[1]/2.)


@typechecked
def create_star_data(path: str,
                     npix_x: int = 100,
                     npix_y: int = 100,
                     x0: List[float] = [50., 50., 50., 50.],
                     y0: List[float] = [50., 50., 50., 50.],
                     parang_start: List[float] = [0., 5., 10., 15.],
                     parang_end: List[float] = [5., 10., 15., 20.],
                     exp_no: List[int] = [1, 2, 3, 4],
                     ndit: int = 10,
                     nframes: int = 10,
                     noise: bool = True) -> None:
    """
    Create data with a stellar PSF and Gaussian noise.

    Parameters
    ----------
    path : str
        Working folder.
    npix_x : int
        Number of pixels in horizontal direction.
    npix_y : int
        Number of pixels in vertical direction.
    x0 : list(float, )
        Positions of the PSF in horizontal direction.
    y0 : list(float, )
        Positions of the PSF in vertical direction.
    parang_start : list(float, )
        Start values of the parallactic angle (deg).
    parang_end : list(float, )
        End values of the parallactic angle (deg).
    exp_no : list(int, )
        Exposure numbers.
    ndit : int
        Number of exposures.
    nframes : int
        Number of frames.
    noise : bool
        Adding noise to the images.

    Returns
    -------
    NoneType
        None
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
            image[i, ] = (1./(2.*np.pi*sigma**2))*np.exp(-((xx-x0[j])**2+(yy-y0[j])**2) /
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
        header['HIERARCH ESO SEQ CUMOFFSETX'] = 'None'
        header['HIERARCH ESO SEQ CUMOFFSETY'] = 'None'
        hdu.data = image
        hdu.writeto(os.path.join(path, 'image'+str(j+1).zfill(2)+'.fits'))


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
    hdu.writeto(os.path.join(path, 'image01.fits'))


@typechecked
def remove_test_data(path: str,
                     folders: List[str] = None,
                     files: List[str] = None) -> None:
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
    image = np.random.normal(loc=0, scale=1., size=(10, 10))

    exp_no = [1, 2, 3, 4]

    for item in exp_no:
        fits_file = os.path.join(path, f'images_'+str(item)+'.fits')

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
