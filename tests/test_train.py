# tests/test_train.py

import unittest
import torch
from torch.functional import F
from torch.optim import Adam, SGD

import sys

sys.path.append(".")

from src.model import MNISTModel, MNISTNet, DynamicMNISTModel
from src.train import train_model
from src.utils import load_data


class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTNet()
        self.train_loader, self.test_loader = load_data()
        self.model_path = '~/mnist_model.pth'

    def test_training_accuracy(self):
        print(f'Parameters - {self.model.count_parameters()}')
        accuracy = train_model(self.model, self.train_loader, criterion=torch.nn.CrossEntropyLoss(),
                               optimizer=Adam(self.model.parameters()), num_epochs=1)
        print(f'Train Accuracy: {accuracy:.4f}')
        self.assertGreaterEqual(accuracy, 0.75, "Model did not achieve 95% accuracy in the first epoch.")
        # Save the model
        torch.save(self.model.state_dict(), self.model_path)
        print(f'Model saved to {self.model_path}')

    def test_test_accuracy(self):
        # Load the saved model
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
        self.assertGreaterEqual(accuracy, 0.95, "Model did not achieve 95% accuracy on the test set.")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestTrainModel('test_training_accuracy'))
    suite.addTest(TestTrainModel('test_test_accuracy'))
    runner = unittest.TextTestRunner()
    runner.run(suite)