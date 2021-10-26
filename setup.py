import setuptools


with open('README.md', 'r') as fin:
    long_description = fin.read()


dependencies = []
with open('requirements.txt') as fin:
    for line in fin:
        dependencies.append(line.strip())


setuptools.setup(
    name='privddnn',
    version='0.1',
    author='Tejas Kannan',
    email='tkannan@uchicago.edu',
    description='Privacy for ML models with Early-Exiting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['privddnn'],
    install_requires=dependencies
)
