from setuptools import setup, find_packages


def get_requirements():
    ''' This function reads the requirements.txt file in current directory 
    and returns those library names as a list of strings here'''

    with open('requirements.txt') as f :
        requirements = [req.replace('\n' , '') for req in f.readlines() if '-e .' not in req]
    
    return requirements

setup(
name = 'bloob',
version = '0.1',
author = 'arun v',
author_email= 'arun_eshwar@yahoo.com',
packages = find_packages(),
install_requires = get_requirements()
)



