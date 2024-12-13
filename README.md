# MNIST PyTorch Project

This project implements a PyTorch-based deep learning model for training on the MNIST dataset. The model is designed to have a maximum of 25,000 parameters and aims to achieve at least 95% training accuracy in the first epoch.

## Project Structure

```
├── src
│   ├── __init__.py          # Marks the src directory as a Python package
│   ├── model.py             # Contains the MNIST model definition
│   ├── train.py             # Handles the training process
│   └── utils.py             # Utility functions for data loading and preprocessing
├── tests
│   ├── __init__.py          # Marks the tests directory as a Python package
│   ├── test_model.py        # Unit tests for the model
│   └── test_train.py        # Unit tests for the training process
├── .github
│   └── workflows
│       └── python-app.yml   # GitHub Actions configuration for CI
├── requirements.txt          # Lists project dependencies
├── README.md                 # Project documentation
└── setup.py                  # Setup script for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/ravipanchani-tomtom/ml-lifecyle-mlops.git
   cd ml-lifecylce-mlops
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```
   python src/train.py
   ```

## Usage

- The model can be trained using the `train_model` function defined in `src/train.py`.
- The MNIST dataset is loaded and preprocessed using utility functions in `src/utils.py`.

## Testing

- Unit tests are provided in the `tests` directory. You can run the tests using:
  ```
  pytest tests/
  ```

## Continuous Integration

This project uses GitHub Actions to run tests automatically on every check-in. The configuration can be found in `.github/workflows/python-app.yml`.