.PHONY: help pypi pypi-test test coverage docs clean clean-build clean-python clean-test

help:
	@echo "pypi - submit to PyPI server"
	@echo "pypi-check - check the distribution for PyPI"
	@echo "pypi-test - submit to TestPyPI server"
	@echo "docs - generate Sphinx documentation"
	@echo "coverage - check code coverage"
	@echo "test - run unit tests"
	@echo "clean - remove artifacts"

pypi:
	python -m build
	twine upload dist/*

pypi-check:
	python -m build
	twine check dist/*

pypi-test:
	python -m build
	twine upload -r testpypi dist/*

docs:
	rm -f docs/pynpoint.core.rst
	rm -f docs/pynpoint.readwrite.rst
	rm -f docs/pynpoint.processing.rst
	rm -f docs/pynpoint.util.rst
	sphinx-apidoc -o docs pynpoint
	cd docs/
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

test:
	pytest --cov=pynpoint/ --cov-report=xml

coverage:
	coverage run --rcfile .coveragerc -m py.test
	coverage combine
	coverage report -m
	coverage html

clean:
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
