from desifm.training.codec_eval import input_style_from_checkpoint


def test_input_style_from_checkpoint():
    assert input_style_from_checkpoint({"input_style": "mask_arcsinh_v3"}) == "mask_arcsinh_v3"
    assert input_style_from_checkpoint({}) == "codec_v2_linear"
    assert input_style_from_checkpoint({"input_style": None}) == "codec_v2_linear"
