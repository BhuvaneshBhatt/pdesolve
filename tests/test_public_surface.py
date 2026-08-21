import pdesolve


def test_public_exports_are_resolvable():
    for name in pdesolve.__all__:
        assert hasattr(pdesolve, name), name


def test_public_api_has_no_compatibility_module():
    assert "compat" not in pdesolve.__all__
