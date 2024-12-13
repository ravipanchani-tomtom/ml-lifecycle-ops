# tests/test_train.py

import unittest
import torch
from torch.functional import F
from torch.optim import Adam, SGD

import sys
sys.path.append(".")

from src.model import MNISTModel
from src.train import train_model
from src.utils import load_data

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
        self.train_loader, self.test_loader = load_data(batch_size=128)

    def test_training_accuracy(self):
        self.model.parameter_summary()
        accuracy = train_model(self.model, self.train_loader,criterion=torch.nn.CrossEntropyLoss(), optimizer=Adam(self.model.parameters(), 0.001), num_epochs=1)
        self.assertGreaterEqual(accuracy, 0.95, "Model did not achieve 95% accuracy in the first epoch.")

if __name__ == '__main__':
    unittest.main()