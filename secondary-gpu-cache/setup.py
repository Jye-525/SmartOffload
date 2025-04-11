from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME
import os


this_dir = os.path.dirname(os.path.abspath(__file__))
cuda_include_dir = os.path.join(CUDA_HOME, "include")

setup(
    name="secondary_gpu_cache",
    ext_modules=[
        CppExtension(
            "secondary_gpu_cache",
            sources=["src/bindings.cpp", "src/secondary_gpu_cache.cpp"],
            include_dirs=[
                os.path.join(this_dir, "include"),
                cuda_include_dir
            ],
            extra_compile_args=["-O3"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)
