"""
Module to obtain information about the implemented attributes.
"""

from typing import Dict, Union

from typeguard import typechecked


@typechecked
def get_attributes() -> Dict[str, Dict[str, Union[str, float, int, None]]]:
    """
    Function to get a dictionary with all attributes.

    Returns
    -------
    dict
        Attribute information.
    """

    attr = {'PIXSCALE': {'attribute': 'static',
                         'config': 'settings',
                         'value': 0.027,
                         'type': 'float'},
            'MEMORY': {'attribute': 'static',
                       'config': 'settings',
                       'value': 1000,
                       'type': 'int'},
            'CPU': {'attribute': 'static',
                    'config': 'settings',
                    'value': 1,
                    'type': 'int'},
            'INSTRUMENT': {'attribute': 'static',
                           'config': 'header',
                           'value': 'INSTRUME',
                           'type': 'str'},
            'NFRAMES': {'attribute': 'non-static',
                        'config': 'header',
                        'value': 'NAXIS3',
                        'type': 'int'},
            'EXP_NO': {'attribute': 'non-static',
                       'config': 'header',
                       'value': 'ESO DET EXP NO',
                       'type': 'int'},
            'DIT': {'attribute': 'static',
                    'config': 'header',
                    'value': 'ESO DET DIT',
                    'type': 'int'},
            'NDIT': {'attribute': 'non-static',
                     'config': 'header',
                     'value': 'ESO DET NDIT',
                     'type': 'int'},
            'PARANG_START': {'attribute': 'non-static',
                             'config': 'header',
                             'value': 'ESO ADA POSANG',
                             'type': 'float'},
            'PARANG_END': {'attribute': 'non-static',
                           'config': 'header',
                           'value': 'ESO ADA POSANG END',
                           'type': 'float'},
            'DITHER_X': {'attribute': 'non-static',
                         'config': 'header',
                         'value': 'ESO SEQ CUMOFFSETX',
                         'type': 'float'},
            'DITHER_Y': {'attribute': 'non-static',
                         'config': 'header',
                         'value': 'ESO SEQ CUMOFFSETY',
                         'type': 'float'},
            'PUPIL': {'attribute': 'non-static',
                      'config': 'header',
                      'value': 'ESO ADA PUPILPOS',
                      'type': 'float'},
            'DATE': {'attribute': 'non-static',
                     'config': 'header',
                     'value': 'DATE-OBS',
                     'type': 'str'},
            'LATITUDE': {'attribute': 'static',
                         'config': 'header',
                         'value': 'ESO TEL GEOLAT',
                         'type': 'float'},
            'LONGITUDE': {'attribute': 'static',
                          'config': 'header',
                          'value': 'ESO TEL GEOLON',
                          'type': 'float'},
            'RA': {'attribute': 'non-static',
                   'config': 'header',
                   'value': 'RA',
                   'type': 'float'},
            'DEC': {'attribute': 'non-static',
                    'config': 'header',
                    'value': 'DEC',
                    'type': 'float'},
            'PARANG': {'attribute': 'non-static',
                       'config': 'header',
                       'value': 'None',
                       'type': 'float'},
            'WAVELENGTH': {'attribute': 'non-static',
                           'config': 'header',
                           'value': 'None',
                           'type': 'float'},
            'STAR_POSITION': {'attribute': 'non-static',
                              'config': None,
                              'value': None,
                              'type': 'float'},
            'INDEX': {'attribute': 'non-static',
                      'config': None,
                      'value': None,
                      'type': 'int'},
            'FILES': {'attribute': 'non-static',
                      'config': None,
                      'value': None,
                      'type': 'str'}}

    return attr
