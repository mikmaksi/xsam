import pytest


def pytest_addoption(parser):
    parser.addoption("--plots", action="store_true", default=False)


@pytest.fixture(scope="class")
def plots_flag(request):
    request.cls.plots_flag = request.config.getoption("--plots")
