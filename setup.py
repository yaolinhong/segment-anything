# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="segment_anything",
    version="1.0",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "fastapi==0.95.0",
        "uvicorn==0.18.3",
        "pydantic==1.10.2",
        "python-multipart==0.0.6",
        # "Pillow==9.2.0",
        "click==8.1.3",
        "scikit-image",
        "open_clip_torch",
    ],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
