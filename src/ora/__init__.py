from importlib.metadata import PackageNotFoundError, version

try:
    from ora._version import __version__
except ModuleNotFoundError:
    # Fallback for source checkouts where setuptools_scm has not generated _version.py.
    try:
        __version__ = version("ora")
    except PackageNotFoundError:
        __version__ = "0.2.0"
