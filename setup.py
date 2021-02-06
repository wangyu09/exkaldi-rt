from setuptools import setup,find_packages
import glob
import os
import subprocess

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#with open("requirement.txt") as fr:
#    requirement = fr.readlines()

def read_version_info():
    cmd = 'cd exkaldi2 && python3 -c "import version; print(version.version.plain)"'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise Exception("Failed to detect ExKaldi Online version.\n"+err.decode())
    else:
        return out.decode().strip().split("\n")[-1].strip()

setup(
    name="exkaldi2",
    version=read_version_info(),
    author="Wang Yu",
    author_email="wangyu@alps-lab.org",
    description="ExKaldi2: An Online Speech Recognition Toolkit for Python",
    long_description=long_description,
    long_description_content_type=os.path.join("text","markdown"),
    url="https://github.com/wangyu09/exkaldi2",
    packages=find_packages(),#["python_scripts",],#find_packages(),
    #data_files = [
    #        (os.path.join("exkaldisrc","tools"), glob.glob( os.path.join("tools","*")))
    #    ],
    install_requires=["numpy>=1.16","PyAudio","webrtcvad"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)