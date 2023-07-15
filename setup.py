from setuptools import setup, find_packages

def requirement_file(path):
    # This function reads the requirements.txt file and returns a list of requirements

    requirements = []
    with open(path) as file_object:
        requirements = file_object.readlines()
        requirements = [package.replace("\n", "") for package in requirements]
        requirements = [package for package in requirements if package != "-e ."]

    return requirements

setup(
    name='Credit Defaults',
    version='0.0.1',
    author='Abhishek',
    author_email='abhishekravikumar24@gmail.com',
    packages=find_packages(),
    install_requires=requirement_file('requirements.txt'),
)
