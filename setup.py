from setuptools import setup, find_packages
from typing import List

def requirement_file(path:str)->List[str]:
    # This function will return list of requirements
    
    requirements = []
    with open(path) as file_object:
        requirements = file_object.readlines()
        requirements = [package.replace("\n", "") for package in requirements]
        if HYPEN_DOT in requirements:
            requirements.remove(HYPEN_DOT)

    return requirements

HYPEN_DOT = '-e .'
setup(
    name='ML_Project',
    version='0.0.1',
    author='Abhishek',
    author_email='abhishekravikumar24@gmail.com',
    packages=find_packages(),
    install_requires=requirement_file('requirement.txt')
)