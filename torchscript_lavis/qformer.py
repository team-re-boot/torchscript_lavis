from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM
from lavis.processors import load_processor
import torch
from pathlib import Path
from torchvision.io import read_image
from lavis.processors.blip_processors import BlipImageBaseProcessor
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from typing import Optional
from PIL import Image
import os
import torchscript_lavis


class Blip2TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, _, self.text_processors = load_model_and_preprocess(
            "blip2_image_text_matching", "pretrain", device=self.device, is_eval=True
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return F.normalize(
            self.model.text_proj(
                self.model.Qformer.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).last_hidden_state[:, 0, :]
            ),
            dim=-1,
        )

    def inference(
        self, text: str, model_filename: str = "qformer_text_encoder.pt"
    ) -> torch.Tensor:
        model = torch.jit.load(model_filename)
        input: torch.Tensor = self.model.tokenizer(
            text,
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        return model(input.input_ids, input.attention_mask)

    def trace(self, model_filename: str = "qformer_text_encoder.pt"):
        trace_tensor: torch.Tensor = self.model.tokenizer(
            "Hello World",
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        traced_model = torch.jit.trace(
            self, (trace_tensor.input_ids, trace_tensor.attention_mask)
        )
        traced_model.save(model_filename)


class Blip2ImageProcessor(BlipImageBaseProcessor):  # type: ignore
    def __init__(
        self,
        image_size: int = 384,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> None:
        super().__init__(mean=mean, std=std)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                self.normalize,
            ]
        )

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        return self.transform(item)


class Blip2ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, _, _ = load_model_and_preprocess(
            "blip2_image_text_matching", "pretrain", device=self.device, is_eval=True
        )
        self.image_processor = Blip2ImageProcessor(image_size=224)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image = self.image_processor(image_tensor)
        assert image.dim() == 3
        image = image.unsqueeze(0).to(self.device)
        with self.model.maybe_autocast():
            image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return F.normalize(
            self.model.vision_proj(query_output.last_hidden_state), dim=-1
        )

    def image_to_tensor(self, image: Image) -> torch.Tensor:
        return transforms.Compose([transforms.ToTensor()])(image)

    def trace(self, model_filename: str = "qformer_image_encoder.pt"):
        image_tensor = self.image_processor(
            self.image_to_tensor(
                Image.open(
                    os.path.join(torchscript_lavis.__path__[0], "merlion_demo.png")
                )
            )[0:3, :, :]
        )
        traced_model = torch.jit.trace(self, (image_tensor))
        traced_model.save(model_filename)


if __name__ == "__main__":
    Blip2TextEncoder().trace()
    Blip2ImageEncoder().trace()
    # Blip2TextEncoder().inference("Hello world, for testing text encoder.")
    pass
