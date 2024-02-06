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
import time
from optimum.bettertransformer import BetterTransformer


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

    def trace_model(
        self,
        model_filename: str = "qformer_text_encoder.pt",
        quantized_model_filename: str = "qformer_text_encoder_quantized.pt",
    ):
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

    def trace_quantized_model(
        self, model_filename: str = "qformer_text_encoder_int8.pt"
    ):
        model_quantized = torch.quantization.quantize_dynamic(
            self, {torch.nn.Linear}, dtype=torch.qint8
        )
        trace_tensor: torch.Tensor = self.model.tokenizer(
            "Hello World",
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        traced_model_quantized = torch.jit.trace(
            model_quantized, (trace_tensor.input_ids, trace_tensor.attention_mask)
        )
        traced_model_quantized.save(model_filename)

    def check_quantize_loss(
        self,
        input_text: str,
        model_filename: str = "qformer_text_encoder.pt",
        quantized_model_filename: str = "qformer_text_encoder_int8.pt",
    ):
        text_embedding_quantized = Blip2TextEncoder().inference(
            input_text,
            model_filename=quantized_model_filename,
        )
        text_embedding = Blip2TextEncoder().inference(
            input_text,
            model_filename=model_filename,
        )
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(text_embedding, text_embedding_quantized)


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
        model, _, _ = load_model_and_preprocess(
            "blip2_image_text_matching", "pretrain", device=self.device, is_eval=True
        )
        self.image_processor = Blip2ImageProcessor(image_size=224)
        self.maybe_autocast = model.maybe_autocast
        self.ln_vision = model.ln_vision
        self.visual_encoder = model.visual_encoder
        self.query_tokens = model.query_tokens
        self.bert = model.Qformer.bert
        self.vision_proj = model.vision_proj

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image = self.image_processor(image_tensor)
        assert image.dim() == 3
        image = image.unsqueeze(0).to(self.device)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

    def inference(
        self, model_filename: str = "qformer_image_encoder.pt"
    ) -> torch.Tensor:
        torchscript_model = torch.jit.load(model_filename)
        image_tensor = self.image_processor(
            self.image_to_tensor(
                Image.open(
                    os.path.join(torchscript_lavis.__path__[0], "merlion_demo.png")
                )
            )[0:3, :, :]
        )
        return torchscript_model(image_tensor)

    def image_to_tensor(self, image: Image) -> torch.Tensor:
        return transforms.Compose([transforms.ToTensor()])(image)

    def trace_model(self, model_filename: str = "qformer_image_encoder.pt"):
        image_tensor = self.image_processor(
            self.image_to_tensor(
                Image.open(
                    os.path.join(torchscript_lavis.__path__[0], "merlion_demo.png")
                )
            )[0:3, :, :]
        )
        # print(image_tensor.size())
        traced_model = torch.jit.trace(self, (image_tensor))
        traced_model.save(model_filename)
        # print(traced_model)

    def trace_quantized_model(
        self, model_filename: str = "qformer_image_encoder_int8.pt"
    ):
        # print(self)
        model_quantized = self
        for i in range(39):
            model_quantized.visual_encoder.blocks[
                i
            ].attn.proj = torch.quantization.quantize_dynamic(
                self.visual_encoder.blocks[i].attn.proj,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            model_quantized.visual_encoder.blocks[
                i
            ].mlp = torch.quantization.quantize_dynamic(
                self.visual_encoder.blocks[i].mlp, {torch.nn.Linear}, dtype=torch.qint8
            )
        model_quantized.bert = torch.quantization.quantize_dynamic(
            self.bert, {torch.nn.Linear}, dtype=torch.qint8
        )
        model_quantized.vision_proj = torch.quantization.quantize_dynamic(
            self.vision_proj, {torch.nn.Linear}, dtype=torch.qint8
        )
        image_tensor = self.image_processor(
            self.image_to_tensor(
                Image.open(
                    os.path.join(torchscript_lavis.__path__[0], "merlion_demo.png")
                )
            )[0:3, :, :]
        )
        print(model_quantized)
        traced_model = torch.jit.trace(model_quantized, (image_tensor))
        traced_model.save(model_filename)

    def optimize_and_save(
        self,
        model_filename: str = "qformer_image_encoder.pt",
        model_out_filename: str = "qformer_image_encoder_optimize.pt",
    ):
        model = torch.jit.load(model_filename)
        torch.jit.optimize_for_inference(torch.jit.script(model.eval())).save(
            model_out_filename
        )

    def check_quantize_loss(
        self,
        model_filename: str = "qformer_image_encoder.pt",
        quantized_model_filename: str = "qformer_image_encoder_int8.pt",
    ):
        embedding_quantized = self.inference(
            model_filename=quantized_model_filename,
        )
        embedding = self.inference(
            model_filename=model_filename,
        )
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        return cos(embedding, embedding_quantized)


if __name__ == "__main__":
    image_encoder = Blip2ImageEncoder()
    # image_encoder.trace_model()
    # image_encoder.trace_quantized_model()
    print(image_encoder.check_quantize_loss())

    # text_encoder = Blip2TextEncoder()
    # text_encoder.trace_quantized_model()
    # print(text_encoder.check_quantize_loss("Hello world, for testing text encoder."))
    # print(text_encoder.check_quantize_loss("I am building robot."))
    # print(
    #     text_encoder.check_quantize_loss(
    #         "The robot is staniding in front of the soccer goal."
    #     )
    # )
    pass
