from setuptools import setup, find_packages

setup(
    name='mnist-pytorch-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A PyTorch project for training a deep learning model on the MNIST dataset.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pytest'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)