from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent
setup(
    name="sglang_grace_cuda",
    ext_modules=[
        CUDAExtension(
            name="grace_cuda._C",
            sources=[
                str(root / "csrc" / "bindings.cpp"),
                str(root / "csrc" / "affinity.cu"),
                str(root / "csrc" / "compute_v2.cu"),
                str(root / "csrc" / "demand.cu"),
                str(root / "csrc" / "placement.cu"),
                str(root / "csrc" / "pure_compute.cu"),
                str(root / "csrc" / "quota.cu"),
                str(root / "csrc" / "traffic.cu"),
                str(root / "csrc" / "runtime.cu"),
            ],
            include_dirs=[str(root / "csrc")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
