from torchscript_lavis.qformer import Blip2ImageEncoder, Blip2TextEncoder


def test_text_encoder():
    Blip2TextEncoder().trace()


def test_image_encoder():
    Blip2ImageEncoder().trace()
