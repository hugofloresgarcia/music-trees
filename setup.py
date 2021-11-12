from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='music_trees',
    description='',
    version='0.0.1',
    install_requires=[
        'pytorch-lightning==1.2.1',
        'nussl',
        'librosa',
        'sox',
        'tqdm',
        'pandas',
        'sklearn',
        'matplotlib',
        'treelib',
        'colorama',
        'natsort',
        'test-tube',
        'str2bool', 
        'numpy==1.20', 
        'ray[tune]',
        'pyyaml<6.0', 
    ],
    packages=['music_trees'],
    package_data={'music_trees': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
