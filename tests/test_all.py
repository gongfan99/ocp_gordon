import pytest
import os

def run_all_tests():
    test_dir = os.path.dirname(__file__)
    pytest.main([test_dir, "-v"])

if __name__ == '__main__':
    run_all_tests()
