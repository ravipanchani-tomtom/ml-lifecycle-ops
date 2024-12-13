# tests/test_model.py

import unittest
import torch
import sys
sys.path.append(".")

from src.model import MNISTModel

class TestMNISTModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()

    def test_parameter_count(self):
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertLessEqual(param_count, 25000, "Model exceeds 25,000 parameters")

    def test_model_architecture(self):
        # Check if the model has the expected layers
        self.assertTrue(hasattr(self.model, 'fc1'), "Model should have a fully connected layer fc1")
        self.assertTrue(hasattr(self.model, 'fc2'), "Model should have a fully connected layer fc2")

if __name__ == '__main__':
    unittest.main()