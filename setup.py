from setuptools import setup


setup(
    name="habet",
    version="0.0.8",
    author="Tom Osika",
    author_email="tom.osika@kitware.com",
    description="The HArmonization BEnchmarking Tool (HABET) streamlines the "
    "process of trying multiple different harmonization techniques and evaluating "
    "their performance through a simple command-line interface (CLI)",
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/KitwareMedical/habet/issues"
    },
    install_requires=[
        "neuroCombat",
        "numpy",
        "pandas",
        "pingouin",
        "itk",
    ],
    entry_points={
        'console_scripts': [
            'habet = habet.main:main',
        ]
    }
)