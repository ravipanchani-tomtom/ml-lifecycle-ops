# tests/test_train.py

import unittest
import torch
from src.model import MNISTModel
from src.train import train_model
from src.utils import load_data

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
        self.train_loader, self.test_loader = load_data()

    def test_training_accuracy(self):
        accuracy = train_model(self.model, self.train_loader, num_epochs=1)
        self.assertGreaterEqual(accuracy, 0.95, "Model did not achieve 95% accuracy in the first epoch.")

if __name__ == '__main__':
    unittest.main()