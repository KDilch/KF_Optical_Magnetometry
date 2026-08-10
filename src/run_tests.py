import unittest
import sys
import os


def run__test(*args):
    # Determine directory of this file (src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Discover tests in tests/
    tests_dir = os.path.join(current_dir, 'tests')
    print(f"Discovering and running tests in: {tests_dir}")

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=tests_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == '__main__':
    run__test()