.. _coding:

Coding a New Module
===================

.. _constructor:

There are three different types of pipeline modules: :class:`~pynpoint.core.processing.ReadingModule`, :class:`~pynpoint.core.processing.WritingModule`, and :class:`~pynpoint.core.processing.ProcessingModule`. The concept is similar for these three modules so here we will explain only how to code a processing module.

Class Constructor
-----------------

First, we need to import the interface (i.e. abstract class) :class:`~pynpoint.core.processing.ProcessingModule`: :

.. code-block:: python

    from pynpoint.core.processing import ProcessingModule

All pipeline modules are classes which contain the parameters of the pipeline step, input ports and/or output ports. So let’s create a simple ``ExampleModule`` class using the ProcessingModule interface (inheritance):

.. code-block:: python

    class ExampleModule(ProcessingModule):

When an IDE like PyCharm is used, a warning will appear that all abstract methods must be implemented in the ``ExampleModule`` class. The abstract class ProcessingModule has some abstract methods which have to be implemented by its children classes (e.g., ``__init__()`` and ``run()``). Thus we have to implement the ``__init__()`` function (i.e., the constructor of our module):

.. code-block:: python

    def __init__(self,
                 name_in="example",
                 in_tag_1="in_tag_1",
                 in_tag_2="in_tag_2",
                 out_tag_1="out_tag_1",
                 out_tag_2="out_tag_2”,
                 parameter_1=0,
                 parameter_2="value"):

Each ``__init__()`` function of a :class:`~pynpoint.core.processing.PypelineModule` requires a ``name_in`` argument (and default value) which is used by the pipeline to run individual modules by name. Furthermore, the input and output tags have to be defined which are used to to access data from the central database. The constructor starts with a call of the :class:`~pynpoint.core.processing.ProcessingModule` interface:

.. code-block:: python
   
    super(ExampleModule, self).__init__(name_in)

Next, the input and output ports behind the database tags have to be defined:

.. code-block:: python

        self.m_in_port_1 = self.add_input_port(in_tag_1)
        self.m_in_port_2 = self.add_input_port(in_tag_2)

        self.m_out_port_1 = self.add_output_port(out_tag_1)
        self.m_out_port_2 = self.add_output_port(out_tag_2)

Reading to and writing from the central database should always be done with the ``add_input_port`` and ``add_output_port`` functionalities and not by manually creating an instance of :class:`~pynpoint.core.dataio.InputPort` or :class:`~pynpoint.core.dataio.OutputPort`.

Finally, the module parameters should be saved to the ``ExampleModule`` instance:

.. code-block:: python

        self.m_parameter_1 = parameter_1
        self.m_parameter_2 = parameter_2

That's it! The constructor of the ``ExampleModule`` is ready.

.. _method:

Run Method
----------

We can now add the functionalities of the module in the ``run()`` method which will be called by the pipeline:

.. code-block:: python

    def run(self):

The input ports of the module are used to load data from the central database into the memory with slicing or the ``get_all()`` function:

.. code-block:: python

        data1 = self.m_in_port_1.get_all()
        data2 = self.m_in_port_2[0:4]

We want to avoid using the ``get_all()`` function because data sets in 3--5 μm range typically consists of thousands of images. Therefore, loading all images at once in the computer memory might not be possible, in particular early in the data reduction chain when the images have their original size. Instead, it is recommended to use the ``MEMORY`` attribute that is specified in the configuration file.

Attributes of the input port are accessed in the following:

.. code-block:: python

        parang = self.m_in_port_1.get_attribute("PARANG")
        pixscale = self.m_in_port_2.get_attribute("PIXSCALE")

And attributes of the central configuration are accessed through the :class:`~pynpoint.core.dataio.ConfigPort`:

.. code-block:: python

        memory = self._m_config_port.get_attribute("MEMORY")
        cpu = self._m_config_port.get_attribute("CPU")

More information on importing of data can be found in the package documentation of :class:`~pynpoint.core.dataio.InputPort`. 

Next, the processing steps are implemented:

.. code-block:: python

        result1 = 10.*self.m_parameter_1
        result2 = 20.*self.m_parameter_1
        result3 = [1, 2, 3]

        attribute = self.m_parameter_2
        
The output ports are used to write the results to the central database:

.. code-block:: python

        self.m_out_port_1.set_all(result1)
        self.m_out_port_1.append(result2)

        self.m_out_port_2[0:2] = result2
        self.m_out_port_2.add_attribute(name="new_attribute", value=attribute)

More information on storing of data can be found in the package documentation of :class:`~pynpoint.core.dataio.OutputPort`.

The attribute information has to be copied from the input port and history information has to be added. This step should be repeated for all the output ports:

.. code-block:: python

        self.m_out_port_1.copy_attributes(self.m_in_port_1)
        self.m_out_port_1.add_history("ExampleModule", "history text")

        self.m_out_port_2.copy_attributes(self.m_in_port_1)
        self.m_out_port_2.add_history("ExampleModule", "history text")

Finally, the central database and all the open ports should be closed:

.. code-block:: python

        self.m_out_port_1.close_port()

.. important::

   It is enough to close only one port because all other ports will be closed automatically.

.. warning::

   It is not recommended to use the same tag name for the input and output port because that would only be possible when data is read and     written at once with the ``get_all()`` and ``set_all()`` functionalities, respectively. Instead image should be read and written in amounts of ``MEMORY`` so an error should be raised when ``in_tag=out_tag``.

.. _example-module:

Example Module
--------------

The full code for the ``ExampleModule`` from above is:

.. code-block:: python

    from pynpoint.core.processing import ProcessingModule

    class ExampleModule(ProcessingModule):

        def __init__(self,
                     name_in="example",
                     in_tag_1="in_tag_1",
                     in_tag_2="in_tag_2",
                     out_tag_1="out_tag_1",
                     out_tag_2="out_tag_2”,
                     parameter_1=0,
                     parameter_2="value"):

            super(ExampleModule, self).__init__(name_in)

            self.m_in_port_1 = self.add_input_port(in_tag_1)
            self.m_in_port_2 = self.add_input_port(in_tag_2)

            self.m_out_port_1 = self.add_output_port(out_tag_1)
            self.m_out_port_2 = self.add_output_port(out_tag_2)

            self.m_parameter_1 = parameter_1
            self.m_parameter_2 = parameter_2

        def run(self):

            data1 = self.m_in_port_1.get_all()
            data2 = self.m_in_port_2[0:4]

            parang = self.m_in_port_1.get_attribute("PARANG")
            pixscale = self.m_in_port_2.get_attribute("PIXSCALE")

            memory = self._m_config_port.get_attribute("MEMORY")
            cpu = self._m_config_port.get_attribute("CPU")

            result1 = 10.*self.m_parameter_1
            result2 = 20.*self.m_parameter_1
            result3 = [1, 2, 3]

            self.m_out_port_1.set_all(result1)
            self.m_out_port_1.append(result2)

            self.m_out_port_2[0:2] = result2
            self.m_out_port_2.add_attribute(name="new_attribute", value=attribute)

            self.m_out_port_1.copy_attributes(self.m_in_port_1)
            self.m_out_port_1.add_history("ExampleModule", "history text")

            self.m_out_port_2.copy_attributes(self.m_in_port_1)
            self.m_out_port_2.add_history("ExampleModule", "history text")

            self.m_out_port_1.close_port()

.. _apply-function:

Apply Function To Images
------------------------

A processing module often applies a specific method to each image of an input port. Therefore, the :func:`~pynpoint.core.processing.ProcessingModule.apply_function_to_images` function has been implemented to apply a function to all images of an input port. This function uses the ``CPU`` and ``MEMORY`` parameter from the configuration file to automatically process subsets of images in parallel. An example of the implementation can be found in the code of the bad pixel cleaning with a sigma filter: :class:`~pynpoint.processing.badpixel.BadPixelSigmaFilterModule`.
