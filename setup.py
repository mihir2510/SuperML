from setuptools import setup, find_packages
import codecs
import os


with open("README.md", "r") as fh:
    long_description = fh.read()


# Setting up
setup(
    name="auto-machine-learning",
    version='0.0.7',
    license='MIT',
    author="Mihir Gada, Zenil Haria, Arnav Mankad, Kaustubh Damania",
    author_email="",
    url = 'https://github.com/mihir2510/AutoML_library',
    download_url ='',
    description='This is an python Library for AutoML which works for prediction and classification tasks.',
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'imblearn',
        'pandas',
        'scikit-optimize',
        'hyperopt',
        'scikit-learn==0.24.1',
        'kiwisolver==1.3.1',
        'matplotlib==3.3.4',
        'Pillow==8.1.0',
        'openpyxl',
        'plotly',
        'pytest',
        'pytest-runner',
        'seaborn',
        'psutil',
        'kaleido',
    ],
    keywords=['automl', 'data preprocessing','feature engineering','ensembling','super learner'],
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
         #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
  ],
  include_package_data=True
)