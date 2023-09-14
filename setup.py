from setuptools import find_packages,setup
from typing import List
import os,time,subprocess,sys

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements


def load_spacy():
    print("Loading spacy dependencies.....")
    
    virtual_env_path = "./venv"
    # activate_script = os.path.join(virtual_env_path, "Scripts", "activate")
    # subprocess.run([activate_script], shell=True, text=True)
    os.environ['PATH'] = f"{virtual_env_path}\\Scripts;{os.environ['PATH']}"
    
    print(subprocess.run([sys.executable, "-m", "spacy", "download", 'en_core_web_lg'], text=True))

    print("Loaded Sucessfully")
    
setup(
    name = "Topic Modeller",
    version= '0.0.1',
    author= 'Manish',
    author_email="manish.rai709130@gmail.com",
    install_requires = get_requirements('requirements.txt'),
    packages= find_packages(),
)
load_spacy()