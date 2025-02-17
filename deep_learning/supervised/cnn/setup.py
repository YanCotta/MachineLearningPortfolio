from setuptools import setup, find_packages

setup(
    name="cnn_image_classifier",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0.0',
        'numpy>=1.19.2',
        'matplotlib>=3.3.2',
        'pillow>=8.0.0',
        'pytest>=6.0.0',
        'typing-extensions>=3.7.4',
    ],
    author="Yan Cotta",
    author_email="your.email@example.com",
    description="A professional CNN implementation for binary image classification",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    keywords="deep learning, CNN, image classification, tensorflow",
    url="https://github.com/yourusername/cnn_image_classifier",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)