"""
Functions for Pypeline modules.
"""

import sys
import time

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
            sys.stdout.write(f'{message} {percentage:4.1f}% - Time: {time_string(time_left)}\r')

    if current+1 == total:
        sys.stdout.write(' ' * (29+len(message)) + '\r')

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
