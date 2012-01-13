from setuptools import setup

version = '0.0.0beta'

long_description = '\n\n'

setup(name='imanu',
      version=version,
      description="IMANU",
      long_description=long_description,
      classifiers=[],
      keywords='IMANU',
      author='Nelle Varoquaux',
      author_email='nelle.varoquaux@gmail.com',
      namespace_packages=[],
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'setuptools',
          # -*- Extra requirements: -*-
      ],
      entry_points={},
      )
