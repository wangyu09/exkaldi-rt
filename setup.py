from setuptools import setup,find_packages
import glob
import os
import subprocess
import importlib

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#with open("requirement.txt") as fr:
#    requirement = fr.readlines()

def read_version_info():
    cmd = 'cd exkaldirt && python3 -c "import version; print(version.version.plain)"'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise Exception("Failed to detect ExKaldi-RT version information.\n"+err.decode())
    else:
        return out.decode().strip().split("\n")[-1].strip()

setup(
    name="exkaldirt",
    version=read_version_info(),
    author="Wang Yu",
    author_email="wangyu@alps-lab.org",
    description="ExKaldi-RT: An Online Speech Recognition Extension Toolkit of Kaldi",
    long_description=long_description,
    long_description_content_type=os.path.join("text","markdown"),
    url="https://github.com/wangyu09/exkaldi-rt",
    packages=find_packages(),#["python_scripts",],
    #data_files = [
    #        (os.path.join("exkaldisrc","tools"), glob.glob( os.path.join("tools","*")))
    #    ],
    install_requires=["numpy>=1.16","PyAudio","webrtcvad","easydict"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
