import unittest
import os
import importlib.util

class TestLLMBench(unittest.TestCase):

    def test_library_torch_installed(self):
        """ Test if torch library is installed """
        torch_installed = importlib.util.find_spec("torch") is not None
        self.assertTrue(torch_installed, "torch library is not installed")

    def test_library_transformers_installed(self):
        """ Test if transformers library is installed """
        transformers_installed = importlib.util.find_spec("transformers") is not None
        self.assertTrue(transformers_installed, "transformers library is not installed")

if __name__ == '__main__':
    unittest.main()