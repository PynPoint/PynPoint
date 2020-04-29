"""
Functions for adding attributes to a dataset in the central database.
"""

import warnings

import numpy as np

from astropy.io import fits
from typeguard import typechecked

from pynpoint.core.attributes import get_attributes
from pynpoint.core.dataio import ConfigPort, OutputPort


@typechecked
def set_static_attr(fits_file: str,
                    header: fits.header.Header,
                    config_port: ConfigPort,
                    image_out_port: OutputPort,
                    check: bool = True) -> None:
    """
    Function which adds the static attributes to the central database.

    Parameters
    ----------
    fits_file : str
        Name of the FITS file.
    header : astropy.io.fits.header.Header
        Header information from the FITS file that is read.
    config_port : pynpoint.core.dataio.ConfigPort
        Configuration port.
    image_out_port : pynpoint.core.dataio.OutputPort
        Output port of the images to which the static attributes are stored.
    check : bool
        Print a warning if certain attributes from the configuration file are not present in
        the FITS header. If set to `False`, attributes are still written to the dataset but
        there will be no warning if a keyword is not found in the FITS header.

    Returns
    -------
    NoneType
        None
    """

    attributes = get_attributes()

    static = []
    for key, value in attributes.items():
        if value['config'] == 'header' and value['attribute'] == 'static':
            static.append(key)

    for attr in static:

        fitskey = config_port.get_attribute(attr)

        if isinstance(fitskey, np.bytes_):
            fitskey = str(fitskey.decode('utf-8'))

        if fitskey != 'None':
            if fitskey in header:
                status = image_out_port.check_static_attribute(attr, header[fitskey])

                if status == 1:
                    image_out_port.add_attribute(attr, header[fitskey], static=True)

                elif status == 0:
                    pass

                elif status == -1:
                    warnings.warn(f'Static attribute {fitskey} has changed. Possibly the '
                                  f'current file {fits_file} does not belong to the data set '
                                  f'\'{image_out_port.tag}\'. Attribute value is updated.')

            elif check:
                warnings.warn(f'Static attribute {attr} (={fitskey}) not found in the FITS '
                              'header.')


@typechecked
def set_nonstatic_attr(header: fits.header.Header,
                       config_port: ConfigPort,
                       image_out_port: OutputPort,
                       check: bool = True) -> None:
    """
    Function which adds the non-static attributes to the central database.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        Header information from the FITS file that is read.
    config_port : pynpoint.core.dataio.ConfigPort
        Configuration port.
    image_out_port : pynpoint.core.dataio.OutputPort
        Output port of the images to which the non-static attributes are stored.

    Returns
    -------
    NoneType
        None
    """

    attributes = get_attributes()

    nonstatic = []
    for key, value in attributes.items():
        if value['attribute'] == 'non-static':
            nonstatic.append(key)

    for attr in nonstatic:
        if attributes[attr]['config'] == 'header':
            fitskey = config_port.get_attribute(attr)

            # if type(fitskey) == np.bytes_:
            #     fitskey = str(fitskey.decode('utf-8'))

            if fitskey != 'None':
                if fitskey in header:
                    image_out_port.append_attribute_data(attr, header[fitskey])

                elif header['NAXIS'] == 2 and attr == 'NFRAMES':
                    image_out_port.append_attribute_data(attr, 1)

                elif check:
                    warnings.warn('Non-static attribute %s (=%s) not found in the '
                                  'FITS header.' % (attr, fitskey))

                    image_out_port.append_attribute_data(attr, -1)


@typechecked
def set_extra_attr(fits_file: str,
                   nimages: int,
                   config_port: ConfigPort,
                   image_out_port: OutputPort,
                   first_index: int) -> int:
    """
    Function which adds extra attributes to the central database.

    Parameters
    ----------
    fits_file : str
        Absolute path and filename of the FITS file.
    nimages : int
        Number of images.
    config_port : pynpoint.core.dataio.ConfigPort
        Configuration port.
    image_out_port : pynpoint.core.dataio.OutputPort
        Output port of the images to which the attributes are stored.
    first_index : int
        First image index of the current subset.

    Returns
    -------
    int
        First image index for the next subset.
    """

    pixscale = config_port.get_attribute('PIXSCALE')

    image_index = np.arange(first_index, first_index+nimages, 1)

    for item in image_index:
        image_out_port.append_attribute_data('INDEX', item)

    image_out_port.append_attribute_data('FILES', fits_file)
    image_out_port.add_attribute('PIXSCALE', pixscale, static=True)

    return first_index + nimages
