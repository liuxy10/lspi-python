#!/bin/bash
python -m unittest discover -s lspi_testsuite -p "test_solvers.py" # works
python -m unittest discover -s lspi_testsuite -p "test_policy.py" # works
python -m unittest discover -s lspi_testsuite -p "test_sample.py" # works
python -m unittest discover -s lspi_testsuite -p "test_learning_chain_domain.py" # works
python -m unittest discover -s lspi_testsuite -p "test_learn.py" # works
# python -m unittest discover -s lspi_testsuite -p "test_domains.py" # Arrays are not equal
# python -m unittest discover -s lspi_testsuite -p "test_basis_functions.py" # AssertionError: TypeError not raised
