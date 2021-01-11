.PHONY: help pypi pypi-test test coverage docs clean clean-build clean-python clean-test

help:
	@echo "pypi - submit to PyPI server"
	@echo "pypi-test - submit to TestPyPI server"
	@echo "docs - generate Sphinx documentation"
	@echo "test - run unit tests"
	@echo "coverage - check code coverage"
	@echo "clean - remove all artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-python - remove Python artifacts"
	@echo "clean-test - remove test artifacts"

pypi:
	python setup.py sdist bdist_wheel
	twine check dist/*
	twine upload dist/*

pypi-test:
	python setup.py sdist bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

docs:
	rm -f docs/pynpoint.core.rst
	rm -f docs/pynpoint.readwrite.rst
	rm -f docs/pynpoint.processing.rst
	rm -f docs/pynpoint.util.rst
	sphinx-apidoc -o docs pynpoint
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

test:
	pytest --cov=pynpoint tests

coverage:
	coverage run --rcfile .coveragerc -m py.test
	coverage combine
	coverage report -m
	coverage html

clean: clean-build clean-python clean-test

clean-build:
	rm -rf dist/
	rm -rf build/
	rm -rf htmlcov/
	rm -rf .eggs/
	rm -rf docs/_build
	rm -rf docs/tutorials/PynPoint_config.ini
	rm -rf docs/tutorials/PynPoint_database.hdf5
	rm -rf docs/tutorials/betapic_naco_mp.hdf5
	rm -rf docs/tutorials/hd142527_zimpol_h-alpha.tgz
	rm -rf docs/tutorials/input
	rm -rf docs/tutorials/.ipynb_checkpoints
	rm -rf docs/tutorials/*.fits
	rm -rf docs/tutorials/*.dat
	rm -rf docs/tutorials/*.npy

clean-python:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -f coverage.xml
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .tox/
	rm -rf pynpoint.egg-info/
	rm -f junit-docs-ci.xml
	rm -f junit-py27.xml
	rm -f junit-py36.xml
	rm -f junit-py37.xml
	rm -rf .pytest_cache/
