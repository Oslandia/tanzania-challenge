
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

def find_version(*file_paths):
    with open(os.path.join(here, *file_paths), 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split('=')[1].strip(' \'"')
    raise RuntimeError(("Unable to find version string. "
                        "Should be __init__.py."))

with open('README.md', 'rb') as f:
    readme = f.read().decode('utf-8')

install_requires = [
    'numpy<=1.14.2',
    'pandas<=0.22.0',
    'pillow<=5.0.0',
    'tensorflow<=1.6',
    'keras<=2.1.5',
    'matplotlib<=2.2.0',
    'luigi<=2.7.8',
    'sh<=1.12.14']

setup(
    name='tanzania challenge',
    keywords=['deep learning', 'convolutional neural networks', 'image', 'Keras'],
    version=find_version('tanzania_challenge', '__init__.py'),
    description='Automatic building detection through instance-specific semantic segmentation',
    long_description=readme,
    license='MIT',
    author='Oslandia',
    author_email='info@oslandia.com',
    maintainer='Oslandia',
    maintainer_email='',
    url='https://github.com/Oslandia/tanzania_challenge',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3',
    install_requires=install_requires,
    packages=find_packages(),
)
