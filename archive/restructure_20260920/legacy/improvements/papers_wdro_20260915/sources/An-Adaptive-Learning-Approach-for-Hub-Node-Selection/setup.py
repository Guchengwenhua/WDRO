from setuptools import Extension, setup


class Pybind11Include:
    """Resolve the pybind11 headers only when the extension is compiled."""

    def __str__(self):
        import pybind11

        return pybind11.get_include()


setup(
    name="adaptive-hub-selection",
    version="0.1.0",
    description="Adaptive Selector-Conductor learning for cost-skew-aware hub selection",
    packages=["models"],
    python_requires=">=3.8",
    ext_modules=[
        Extension(
            "models._mspd",
            ["cpp/mspd.cpp"],
            include_dirs=[Pybind11Include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++14"],
        )
    ]
)
