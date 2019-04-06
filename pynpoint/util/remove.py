"""
Functions to write selected data and attributes to the central database.
"""

from __future__ import absolute_import

import numpy as np


def write_selected_data(images,
                        indices,
                        port_selected,
                        port_removed):
    """
    Function to write a selected number of images from a data set.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of images.
    indices : numpy.ndarray
        Indices that are removed.
    port_selected : pynpoint.core.dataio.OutputPort
        Port to store the selected images.
    port_removed : pynpoint.core.dataio.OutputPort
        Port to store the removed images.

    Returns
    -------
    NoneType
        None
    """

    if np.size(indices) > 0:
        if port_removed is not None:
            port_removed.append(images[indices])

        if images.ndim == 2:
            images = None

        elif images.ndim == 3:
            images = np.delete(images, indices, axis=0)

    if port_selected is not None and images is not None:
        port_selected.append(images)

def write_selected_attributes(indices,
                              port_input,
                              port_selected,
                              port_removed):
    """
    Function to write the attributes of a selected number of images.

    Parameters
    ----------
    indices : numpy.ndarray
        Indices that are removed.
    port_input : pynpoint.core.dataio.InputPort
        Port to the input data.
    port_selected : pynpoint.core.dataio.OutputPort
        Port to store the attributes of the selected images.
    port_removed : pynpoint.core.dataio.OutputPort
        Port to store the attributes of the removed images. Not written if set to None.

    Returns
    -------
    NoneType
        None
    """

    non_static = port_input.get_all_non_static_attributes()

    for i, item in enumerate(non_static):
        values = port_input.get_attribute(item)

        if values.shape[0] == port_input.get_shape()[0]:

            if port_selected is not None:
                if np.size(indices) > 0:
                    if values.ndim == 1:
                        selected = np.delete(values, indices)

                    elif values.ndim == 2:
                        selected = np.delete(values, indices, axis=0)

                else:
                    selected = values

                port_selected.add_attribute(item, selected, static=False)

            if port_removed is not None and np.size(indices) > 0:
                removed = values[indices]

                port_removed.add_attribute(item, removed, static=False)

    if "NFRAMES" in non_static:
        nframes = port_input.get_attribute("NFRAMES")

        nframes_sel = np.zeros(nframes.shape, dtype=np.int)
        nframes_del = np.zeros(nframes.shape, dtype=np.int)

        for i, frames in enumerate(nframes):
            total = np.sum(nframes[0:i])

            if np.size(indices) > 0:
                index_del = np.where(np.logical_and(indices >= total, \
                                     indices < total+frames))[0]

                nframes_sel[i] = frames-np.size(index_del)
                nframes_del[i] = np.size(index_del)

            else:
                nframes_sel[i] = frames
                nframes_del[i] = 0

        if port_selected is not None:
            port_selected.add_attribute("NFRAMES", nframes_sel, static=False)

        if port_removed is not None:
            port_removed.add_attribute("NFRAMES", nframes_del, static=False)
