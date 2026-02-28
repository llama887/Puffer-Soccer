import pytest


def test_pufferl_smoke_imports():
    pufferlib = pytest.importorskip("pufferlib")
    assert pufferlib.__version__ >= 3.0
