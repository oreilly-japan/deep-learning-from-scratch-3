from setuptools import setup
from dezero import __version__

setup(name='dezero',
      version=__version__,
      license='MIT License',
      install_requires=['numpy'],
      description='Deep Learning Framework from Zero',
      author='Koki Saitoh',
      author_email='koki0702@gmail.com',
      url='',
      packages=['dezero'],
      )