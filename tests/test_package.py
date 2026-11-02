import semcache


def test_version_matches_the_package_metadata():
    from importlib.metadata import version

    assert semcache.__version__ == version("semcache")
