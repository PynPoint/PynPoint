"""
Functions for the test cases.
"""

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
