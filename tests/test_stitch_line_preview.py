import numpy as np

from experiments.stitch_line_preview import LayoutConfig, layout


def test_initial_layouts_have_no_mixing_and_collinear_stitches():
    config = LayoutConfig()
    for index in range(4):
        supports, stitches, center, normal = layout(config, index)
        assert supports.shape == (2, 28, 2)
        assert (((supports[0] - center) @ normal) < 0).all()
        assert (((supports[1] - center) @ normal) > 0).all()
        np.testing.assert_allclose((stitches - center) @ normal, 0, atol=1e-14)
        assert ((stitches > 0) & (stitches < 1)).all()
