from setuptools import setup

with open ("README.md","r",encoding="utf-16") as fh:
    long_description = fh.read()

AUTHOR_NAME = '999'
SRC_REPO = 'src'
LIST_OF_REQUIRMENTS = ['streamlit']

setup(
    name=SRC_REPO,
    version= '0.0.1',
    author=AUTHOR_NAME,
    author_email='mustansirrashid@hotmail.com',
    description='No need to brainstorm to watch your next movie',
    long_description= long_description,
    long_description_content_type= 'text/markdown',
    packages= [SRC_REPO],
    python_requires = '>=3.11',
    install_requires = LIST_OF_REQUIRMENTS
)