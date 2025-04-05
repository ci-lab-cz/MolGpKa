from setuptools import setup, find_packages

setup(
    name='molgpka',
    version='1.1',
    author='',
    author_email='',
    description='A tool for pKa prediction using graph-convolutional neural network models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ci-lab-cz/MolGpKa',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        # 'torch>=1.4',
        # 'torch-geometric',
        # 'rdkit',
        # Add other dependencies from environment.yml if necessary
    ],
)
