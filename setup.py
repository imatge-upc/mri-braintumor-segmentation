from setuptools import setup, find_packages

setup(name='BrainTumorSegmentation',
      description="",
      version='1.0.0',
      author='LauraMora',
      packages=find_packages(),
      project_urls={"Source Code": ""},
      install_requires=[line.rstrip('\n') for line in open('requirements.txt') if '--' not in line]
      )
