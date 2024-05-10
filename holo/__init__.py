from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("holo-bench")
except PackageNotFoundError:
    __version__ = "unknown version"
