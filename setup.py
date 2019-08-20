
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
    'configparser<=3.5.0',
    'keras<=2.2.2',
    'numpy<=1.17',
    'pandas<=0.22.0',
    'luigi<=2.8.8',
    'Pillow<=5.0.0',
    'requests<=2.19.1'
    'sh<=1.12.14',
    'tensorflow<=1.12.2']

dependency_links = [
    'git+https://github.com/mapnik/python-mapnik@v3.0.16',
    'git+https://github.com/matterport/Mask_RCNN',
]

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
    dependency_links=dependency_links,
    packages=find_packages(),
)
