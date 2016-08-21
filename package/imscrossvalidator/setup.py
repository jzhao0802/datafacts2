from setuptools import setup

setup(name='imscrossvalidator',
      version='0.1',
      description='CrossValidator with custom stratification (matching or grouping patients of same category in same fold)',
      author='Aditya Konda',
      author_email='akonda@uk.imshealth.com',
      packages=['crossvalidator','stratification'],
      zip_safe=False)
