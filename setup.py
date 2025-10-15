"""Setup script for pyCAFE"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read version from package
def get_version():
    version = {}
    with open(os.path.join('pyCAFE', '__init__.py'), 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                exec(line, version)
                return version['__version__']
    return '0.1.0'

setup(
    name='pyCAFE',
    version=get_version(),
    author='Wulin Tan',
    author_email='wulintan9527@gmail.com',
    description='Python CUDA Accelerated Frame Extractor - GPU-accelerated video frame extraction with K-means clustering',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/Wulin-Tan/pyCAFE',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Video',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
        'Pillow>=8.0.0',
        'tqdm>=4.60.0',
        'scikit-learn>=0.24.0',
    ],
    extras_require={
        'gpu': [
            'cupy-cuda11x>=10.0.0',  # Adjust CUDA version as needed
            'cuml-cu11>=22.0.0',     # Adjust CUDA version as needed
            'nvidia-dali-cuda110>=1.20.0',  # Adjust CUDA version as needed
        ],
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
        ],
    },
    entry_points={
        'console_scripts': [
            'pyCAFE=pyCAFE.cli:main',
        ],
    },
    include_package_data=True,
    keywords='video frame-extraction gpu cuda kmeans clustering dali cuml',
    project_urls={
        'Bug Reports': 'https://github.com/Wulin-Tan/pyCAFE/issues',
        'Source': 'https://github.com/Wulin-Tan/pyCAFE',
        'Documentation': 'https://github.com/Wulin-Tan/pyCAFE#readme',
    },
)
