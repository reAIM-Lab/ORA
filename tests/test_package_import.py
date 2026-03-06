import ora


def test_package_imports() -> None:
    assert ora.__name__ == "ora"


def test_version_is_exposed() -> None:
    assert isinstance(ora.__version__, str)
    assert ora.__version__
