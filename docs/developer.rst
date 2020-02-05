=======================
Developer Documentation
=======================
This project is an open-source community project,
hosted on GitHub at the following address:
https://github.com/FenTechSolutions/CausalDiscoveryToolbox

We abide by the principles of openness, respect, and consideration of others of
the Python Software Foundation: https://www.python.org/psf/codeofconduct/


Bug reporting
=============
Encountering a bug while using this package may occur. In order to fix the said
bug and improve all users' experience, it is highly recommended to submit a bug
report on the GitHub issue tracker: https://github.com/FenTechSolutions/CausalDiscoveryToolbox/issues

When reporting a bug, please mention:

- Your ``cdt`` package version or docker image tag.

- Your python version.

- Your ``PyTorch`` package version.

- Your hardware configuration, if there are GPUs available.

- The full traceback of the raised error if one is raised.

- A small code snippet to reproduce the bug if the description is not explicit.

Contributing
============
The recommended way to contribute to the Causal Discovery Toolbox is to submit a
pull request on the ``dev`` branch of https://github.com/FenTechSolutions/CausalDiscoveryToolbox

To submit a pull request, the following are required:

1. Having an up-to-date forked repository of the package and a python 3 installation

2. Clone your forked version of the code locally and install it
   in developer mode, in a separate python environement
   (e.g. Anaconda environement)::

       $ conda create --name cdt_dev python=3.6 numpy scipy scikit-learn
       $ source activate cdt_dev
       $ git clone git@github.com:YourLogin/CausalDiscoveryToolbox.git
       $ cd CausalDiscoveryToolbox
       $ git checkout dev
       $ python setup.py install develop

   Where ``python`` refers to your `python 3` installation.

3. Make your changes to the source code of the package

4. Test your changes using ``pytest``::

       $ cd CausalDiscoveryToolbox
       $ pip install pytest
       $ pytest

5. If the tests pass, commit and push your changes::

       $ git add .
       $ git commit -m "[DEV] Your commit message"
       $ git push -u origin dev

   The commits must begin with a tag, defining the main purpose of the commit.
   Examples of tags are:

   - ``[DEV]`` for development

   - ``[TRAVIS]`` for changes on the continuous integration

   - ``[DOC]`` for documentation

   - ``[TEST]`` for testing and coverage

   - ``[FIX]`` for bugfixes

   - ``[REL]`` and ``[MREL]`` are reserved names for releases and major releases.
     They trigger package version updates on the continuous integration.

   - ``[DEPLOY]`` is a reserved tag for the continuous integration to upload
     its changes.


6. Please check that your pull request complies with all the rules of the checklist:

   - Respected the pattern design of the package, using the ``networkx.DiGraph``
     classes and the ``cdt.Settings`` modules and heritage from the model classes,
     and verified the correct import of the new functionalities.

   - Added documentation to your added functionalities (check the following section)

   - Added corresponding tests to the added functions/classes in ``/tests/scripts``

7. Finally, submit your pull request using the GitHub website.


Dependencies
============
The package is to be as much independent of other packages as possible, as it
already depends on many libraries. Therefore, all contributions requiring
the addition of a new dependency will be severely examined.

Two types of dependencies are possible for now:

- Python dependencies, defined in ``requirements.txt`` and ``setup.py``

- R dependencies, defined in ``r_requirements.txt``

.. warning::
   For R dependencies, the Docker base images have to be rebuilt, thus notifying
   the core maintainers of the package is necessary for the Docker image to be
   updated.

Documentation
=============
The documentation of the package is automatically generated using `Sphinx`, by
parsing docstrings of functions and classes, as defined in ``/docs/index.md``
and the ``/docs/*.rst`` files. To add a new function in the documentation, add
the respective mention in the ``.rst`` file. The documentation is automatically
built and updated online by the Continuous Integration Tool at each push on the
`master` branch.

When writing your docstrings, please use the Google Style format:
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Your docstrings must include:

- A presentation of the functionality

- A detailed description or the arguments and returns

- A scientific source in ``..note::`` if applicable

- A short example

Testing
=======
The package is thoroughly tested using ``pytest`` and ``codecov`` for code
coverage. Tests are run using a Continuous Integration Tool, for
each push on master/dev or pull requests, allowing to provide users with
immediate feedback.

The test scripts are included in the GitHub repository at ``/tests/scripts``,
and some sample data for the function to be applied on can be found in
``/tests/datasets``.

In order to write new tests functions, add either a new python file or complete
an already existing file, and add a function whose name must begin with ``test_``.
This allows pytest to automatically detect the new test function.

New test functions must provide optimal code coverage of tested functionalities,
as well as test of imports and result coherence.

Continuous Integration
======================
Continuous integration (travis-ci) is enabled on this project, it allows for:

1. Testing new code with ``pytest`` and upload the code coverage results to https://codecov.io/gh/FenTechSolutions/CausalDiscoveryToolbox

2. Bumping a new version of the package and push it to GitHub.

3. Building new docker images and push them to https://hub.docker.com/u/fentech

4. Push the new package version to PyPi

5. Compile the new documentation and upload its website.

All the tasks described above are defined in the ``.travis.yml`` file.

R integration
=============

One of this project's main features is wrapping around R-libraries. In order to
do it in the most efficient way, the `R` tasks are executed in a different process
than the main `python` process thus freeing the computation from the GIL.

A `/tmp/` folder is used as buffer, and everything is executed with the
subprocess library. Check out :ref:`cdt.utils.R` for more
detailed information.

Parallelization
===============

Many algorithms are computationally heavy, but parallelizable, as they include
bootstrapped functions, multiple runs of a same computation.

Therefore, using multiprocessing allows to alleviate the required computation
time. For CPU jobs, we use the ``joblib`` library, for its efficiency and ease
of use. However, for GPU jobs, the multiprocessing interface was recoded,
in order to account for available resources and a memory leak issue between
`joblib` and `PyTorch`.

Check out :ref:`cdt.utils.parallel` for more detailed information.
