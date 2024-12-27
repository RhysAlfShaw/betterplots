import betterstyle as bs

# test adding betterstyle style


def test_init_style():
    bs.set_style("betterstyle")
    assert bs.get_style() == "betterstyle"
