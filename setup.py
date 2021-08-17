import setuptools


with open('README.md', 'r') as fin:
    long_description = fin.read()


setuptools.setup(
    name='privddnn',
    version='0.1',
    author='Tejas Kannan',
    email='tkannan@uchicago.edu',
    description='Privacy for Deep Neural Networks with Early-Exiting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages()
)
