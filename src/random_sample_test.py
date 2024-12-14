import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def save_random_examples(model, data_loader, log_dir='logs', num_examples=5):
    os.makedirs(log_dir, exist_ok=True)
    model.eval()
    examples = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(images)):
                if len(examples) < num_examples:
                    examples.append((images[i], labels[i], predicted[i]))
                else:
                    break
            if len(examples) >= num_examples:
                break

    for i, (image, label, prediction) in enumerate(examples):
        save_image(image, os.path.join(log_dir, f'example_{i}.png'))
        with open(os.path.join(log_dir, f'example_{i}.txt'), 'w') as f:
            f.write(f'Label: {label.item()}, Prediction: {prediction.item()}')

# Example usage
if __name__ == "__main__":
    from model import MNISTNet
    from utils import load_data

    model = MNISTNet()
    model.load_state_dict(torch.load('mnist_model.pth'))
    train_loader, test_loader = load_data()
    save_random_examples(model, test_loader)