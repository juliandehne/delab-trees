
import unittest
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Discover and run all tests in the directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover(current_dir, pattern='*_tests.py')
test_runner = unittest.TextTestRunner()
test_runner.run(test_suite)