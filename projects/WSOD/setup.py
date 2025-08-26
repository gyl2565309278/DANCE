#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import os
from os import path
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "wsod", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )
    if is_rocm_pytorch:
        assert torch_ver >= [1, 8], "ROCM support requires PyTorch >= 1.8!"

    # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu"))
    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            extra_compile_args["nvcc"].extend(nvcc_flags_env.split(" "))

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    # DenseCRF
    crf_sources = glob.glob(path.join(extensions_dir, "crf", "densecrf", "src", "*.cpp"))
    crf_include_dirs = path.join(extensions_dir, "crf", "densecrf", "include")
    for source in crf_sources:
        if "optimization" in source:
            continue
        if "examples" in source:
            continue
        sources += [source]
    include_dirs += [crf_include_dirs]

    ext_modules = [
        extension(
            "wsod._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="wsod",
    version="1.0",
    author="Yuelin Guo",
    url="https://github.com/gyl2565309278/detectron2",
    description="A research platform for weakly supervised object detection based on Detectron2.",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.7",
    install_requires=["graphviz"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
