from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='torch_efficient_distloss',
    packages=['torch_efficient_distloss'],
    version='0.1',
    license='MIT',
    description='Efficient distortion loss with O(n) realization.',
    long_description=long_description,
    author='Cheng Sun',
    author_email='chengsun@gapp.nthu.edu.tw',
    url='https://github.com/sunset1995',
    download_url='https://github.com/sunset1995/py360convert/archive/v_0.1.0.tar.gz',
    install_requires=[],
)

