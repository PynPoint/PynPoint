"""
Functions for Pypeline modules.
"""

import sys
import numpy as np


def progress(current, total, message):
    """
    Function to show and update the progress as standard output.
    """

    fraction = float(current)/float(total)
    percentage = round(fraction*100., 1)

    sys.stdout.write("%s %s%s \r" % (message, percentage, "%"))
    sys.stdout.flush()

def memory_frames(memory, nimages):
    """
    Function to subdivide the input images is in quantities of MEMORY.
    """

    if memory == 0 or memory >= nimages:
        frames = [0, nimages]

    else:
        frames = np.linspace(0,
                             nimages-nimages%memory,
                             int(float(nimages)/float(memory))+1,
                             endpoint=True,
                             dtype=np.int)

        if nimages%memory > 0:
            frames = np.append(frames, nimages)

    return frames

def number_images(port):
    """
    Function to get the number of images of an input port.
    """

    if port.get_ndim() == 2:
        nimages = 1

    elif port.get_ndim() == 3:
        nimages = port.get_shape()[0]

    return nimages
