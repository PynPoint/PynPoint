"""
Functions to write selected data and attributes to the central database.
"""

import time

from typing import Union

import numpy as np

from typeguard import typechecked

from pynpoint.core.dataio import InputPort, OutputPort
from pynpoint.util.module import progress, memory_frames


@typechecked
def write_selected_data(memory: Union[int, np.int64],
                        indices: np.ndarray,
                        image_in_port: InputPort,
                        selected_out_port: OutputPort = None,
                        removed_out_port: OutputPort = None) -> None:
    """
    Function to write a selected number of images from a data set.

    Parameters
    ----------
    memory : int
        Number of images that is simultaneously loaded into the memory
    indices : numpy.ndarray
        Image indices that will be removed.
    image_in_port : pynpoint.core.dataio.InputPort
        Port to the input images.
    selected_out_port : pynpoint.core.dataio.OutputPort, None
        Port to store the selected images.
    removed_out_port : pynpoint.core.dataio.OutputPort, None
        Port to store the removed images.

    Returns
    -------
    NoneType
        None
    """

    nimages = image_in_port.get_shape()[0]
    frames = memory_frames(memory, nimages)

    if memory == 0 or memory >= nimages:
        memory = nimages

    start_time = time.time()

    for i, _ in enumerate(frames[:-1]):
        progress(i, len(frames[:-1]), 'Writing selected data...', start_time)

        images = image_in_port[frames[i]:frames[i+1], ]

        subset_del = np.where(np.logical_and(indices >= frames[i],
                                             indices < frames[i+1]))[0]

        index_del = indices[subset_del] % memory

        index_sel = np.ones(images.shape[0], np.bool)
        index_sel[index_del] = 0

        if selected_out_port is not None and index_sel.size > 0:
            selected_out_port.append(images[index_sel])

        if removed_out_port is not None and indices.size != images.shape[0]:
            removed_out_port.append(images[index_del])


@typechecked
def write_selected_attributes(indices: np.ndarray,
                              image_in_port: InputPort,
                              selected_out_port: OutputPort = None,
                              removed_out_port: OutputPort = None) -> None:
    """
    Function to write the attributes of a selected number of images.

    Parameters
    ----------
    indices : numpy.ndarray
        Image indices that will be removed.
    image_in_port : pynpoint.core.dataio.InputPort
        Port to the input data.
    selected_out_port : pynpoint.core.dataio.OutputPort, None
        Port to store the attributes of the selected images. Not written if set to None.
    removed_out_port : pynpoint.core.dataio.OutputPort, None
        Port to store the attributes of the removed images. Not written if set to None.

    Returns
    -------
    NoneType
        None
    """

    non_static = image_in_port.get_all_non_static_attributes()

    index_sel = np.ones(image_in_port.get_shape()[0], np.bool)
    index_sel[indices] = 0

    for i, attr_item in enumerate(non_static):
        values = image_in_port.get_attribute(attr_item)

        if values.shape[0] == image_in_port.get_shape()[0]:

            if selected_out_port is not None and index_sel.size > 0:
                selected_out_port.add_attribute(attr_item, values[index_sel], static=False)

            if removed_out_port is not None and indices.size > 0:
                removed_out_port.add_attribute(attr_item, values[indices], static=False)

    if 'NFRAMES' in non_static:
        nframes = image_in_port.get_attribute('NFRAMES')

        nframes_sel = np.zeros(nframes.shape, dtype=np.int)
        nframes_del = np.zeros(nframes.shape, dtype=np.int)

        for i, frames in enumerate(nframes):
            total = np.sum(nframes[:i])

            if indices.size == 0:
                nframes_sel[i] = frames
                nframes_del[i] = 0

            else:
                index_del = np.where(np.logical_and(indices >= total,
                                                    indices < total+frames))[0]

                nframes_sel[i] = frames - index_del.size
                nframes_del[i] = index_del.size

        if selected_out_port is not None:
            selected_out_port.add_attribute('NFRAMES', nframes_sel, static=False)

        if removed_out_port is not None:
            removed_out_port.add_attribute('NFRAMES', nframes_del, static=False)
