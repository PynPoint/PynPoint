"""
Functions for Pypeline modules.
"""

import sys
import time
import math
import cmath
import warnings

from typing import Union

import numpy as np

from typeguard import typechecked


@typechecked
def progress(current: int,
             total: int,
             message: str,
             start_time: float = None) -> None:
    """
    Function to show and update the progress as standard output.

    Parameters
    ----------
    current : int
        Current index.
    total : int
        Total index number.
    message : str
        Message that is printed.
    start_time : float, None, optional
        Start time in seconds. Not used if set to None.

    Returns
    -------
    NoneType
        None
    """

    def time_string(delta_time):
        """
        Converts to input time in seconds to a string which displays as hh:mm:ss.

        Parameters
        ----------
        delta_time : float
            Input time in seconds.

        Returns
        -------
        str:
            String with the formatted time.
        """

        hours = int(delta_time / 3600.)
        minutes = int((delta_time % 3600.) / 60.)
        seconds = int(delta_time % 60.)

        return f'{hours:>02}:{minutes:>02}:{seconds:>02}'

    fraction = float(current) / float(total)
    percentage = 100.*fraction

    if start_time is None:
        sys.stdout.write(f'\r{message} {percentage:4.1f}% \r')

    else:
        if fraction > 0. and current+1 != total:
            time_taken = time.time() - start_time
            time_left = time_taken / fraction * (1. - fraction)
            sys.stdout.write(f'{message} {percentage:4.1f}% - ETA: {time_string(time_left)}\r')

    if current+1 == total:
        sys.stdout.write((29 + len(message)) * ' ' + '\r')
        sys.stdout.write(message+' [DONE]\n')

    sys.stdout.flush()


@typechecked
def memory_frames(memory: Union[int, np.int64],
                  nimages: int) -> np.ndarray:
    """
    Function to subdivide the input images is in quantities of MEMORY.

    Parameters
    ----------
    memory : int
        Number of images that is simultaneously loaded into the memory.
    nimages : int
        Number of images in the stack.

    Returns
    -------
    numpy.ndarray
        Array with the indices where a stack of images is subdivided.
    """

    if memory == 0 or memory >= nimages:
        frames = np.asarray([0, nimages])

    else:
        frames = np.linspace(start=0,
                             stop=nimages - nimages % memory,
                             num=int(float(nimages)/float(memory))+1,
                             endpoint=True,
                             dtype=np.int)

        if nimages % memory > 0:
            frames = np.append(frames, nimages)

    return frames


@typechecked
def angle_average(angles: np.ndarray) -> float:
    """
    Function to calculate the average value of a list of angles.

    Parameters
    ----------
    angles : numpy.ndarray
        Parallactic angles (deg).

    Returns
    -------
    float
        Average angle (deg).
    """

    cmath_rect = sum(cmath.rect(1, math.radians(ang)) for ang in angles)
    cmath_phase = cmath.phase(cmath_rect/len(angles))

    return math.degrees(cmath_phase)


@typechecked
def angle_difference(angle_1: float,
                     angle_2: float) -> float:
    """
    Function to calculate the difference between two  angles.

    Parameters
    ----------
    angle_1 : float
        First angle (deg).
    angle_2 : float
        Second angle (deg).

    Returns
    -------
    float
        Angle difference (deg).
    """

    angle_diff = (angle_1-angle_2) % 360.

    if angle_diff >= 180.:
        angle_diff -= 360.

    return angle_diff


@typechecked
def stack_angles(memory: Union[int, np.int64],
                 parang: np.ndarray,
                 max_rotation: float) -> np.ndarray:
    """
    Function to subdivide the input images is in quantities of MEMORY with a restriction on the
    maximum field rotation across a subset of images.

    Parameters
    ----------
    memory : int
        Number of images that is simultaneously loaded into the memory.
    parang : numpy.ndarray
        Parallactic angles (deg).
    max_rotation : float
        Maximum field rotation (deg).

    Returns
    -------
    numpy.ndarray
        Array with the indices where a stack of images is subdivided.
    """

    warnings.warn('Testing of util.module.stack_angles has been limited, please use carefully.')

    nimages = parang.size

    if memory == 0 or memory >= nimages:
        frames = [0, nimages]

    else:
        frames = [0, ]
        parang_start = parang[0]
        im_count = 0

        for i in range(1, parang.size):
            abs_start_diff = abs(angle_difference(parang_start, parang[i-1]))
            abs_current_diff = abs(angle_difference(parang[i], parang[i-1]))

            if abs_start_diff > max_rotation or abs_current_diff > max_rotation:
                frames.append(i)
                parang_start = parang[i]
                im_count = 0

            else:
                im_count += 1

                if im_count == memory:
                    frames.append(i)

                    if i < parang.size-1:
                        parang_start = parang[i+1]
                        im_count = 0

        if frames[-1] != nimages:
            frames.append(nimages)

    return np.asarray(frames)


@typechecked
def update_arguments(index: int,
                     nimages: int,
                     args_in: Union[tuple, None]) -> Union[tuple, None]:
    """
    Function to update the arguments of an input function. Specifically, arguments which contain an
    array with the first dimension equal in size to the total number of images will be substituted
    by the array element of the image index.

    Parameters
    ----------
    index : int
        Image index in the stack.
    nimages : int
        Total number of images in the stack.
    args_in : tuple, None
        Function arguments that have to be updated.

    Returns
    -------
    tuple, None
        Updated function arguments.
    """

    if args_in is None:
        args_out = None

    else:
        args_out = []

        for item in args_in:
            if isinstance(item, np.ndarray) and item.shape[0] == nimages:
                args_out.append(item[index])

            else:
                args_out.append(item)

        args_out = tuple(args_out)

    return args_out


@typechecked
def module_info(pipeline_module) -> None:
    """
    Function to print the module name.

    Parameters
    ----------
    pipeline_module : PypelineModule
        Pipeline module.

    Returns
    -------
    NoneType
        None
    """

    module_name = type(pipeline_module).__name__
    str_length = len(module_name)

    print('\n' + str_length * '-')
    print(module_name)
    print(str_length * '-' + '\n')
    print(f'Module name: {pipeline_module._m_name}')


@typechecked
def input_info(pipeline_module) -> None:
    """
    Function to print information about the input data.

    Parameters
    ----------
    pipeline_module : PypelineModule
        Pipeline module.

    Returns
    -------
    NoneType
        None
    """

    input_ports = list(pipeline_module._m_input_ports.keys())

    if len(input_ports) == 1:
        input_shape = pipeline_module._m_input_ports[input_ports[0]].get_shape()
        print(f'Input port: {input_ports[0]} {input_shape}')

    else:
        print('Input ports:', end='')

        for i, item in enumerate(input_ports):
            input_shape = pipeline_module._m_input_ports[input_ports[i]].get_shape()

            if i < len(input_ports) - 1:
                print(f' {item} {input_shape},', end='')
            else:
                print(f' {item} {input_shape}')


@typechecked
def output_info(pipeline_module, output_shape) -> None:
    """
    Function to print information about the output data.

    Parameters
    ----------
    pipeline_module : PypelineModule
        Pipeline module.
    output_shape : dict
        Dictionary with the output dataset names and shapes.

    Returns
    -------
    NoneType
        None
    """

    output_ports = list(pipeline_module._m_output_ports.keys())

    if len(output_ports) == 1:
        if output_ports[0][:11] != 'fits_header':
            print(f'Output port: {output_ports[0]} {output_shape[output_ports[0]]}')

    else:
        print('Output ports:', end='')

        for i, item in enumerate(output_ports):
            if i < len(output_ports) - 1:
                print(f' {item} {output_shape[output_ports[i]]},', end='')
            else:
                print(f' {item} {output_shape[output_ports[i]]}')
