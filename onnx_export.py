import torch
from torch import nn
import numpy as np
import clip
from PIL import Image


class ClipDecoder(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logits_per_image = 100 * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image.softmax(1), logits_per_text.softmax(1)


device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "iverify", "a cat"]).to(device)

decoder = ClipDecoder()

with torch.no_grad():
    image_features = model.encode_image(image)

    torch.onnx.export(model.visual,
                      image,
                      "image_encoder.onnx",
                      input_names=("images", ),
                      output_names=("image_features", ),
                      dynamic_axes={"images": {
                          0: "num_image"
                      }})
    # text_features = model.encode_text(text)

    text_features = model(text)

    torch.onnx.export(model, (text, ),
                      "text_encoder.onnx",
                      input_names=("texts", ),
                      output_names=("text_features", ),
                      dynamic_axes={"texts": {
                          0: "num_text"
                      }})
    logits_per_image, logits_per_text = decoder(image_features, text_features)

    torch.onnx.export(decoder, (image_features, text_features),
                      "feature_matmul.onnx",
                      input_names=("image_features", "text_features"),
                      output_names=("logits_per_image", "logits_per_text"),
                      dynamic_axes={
                          "image_features": {
                              0: "num_image"
                          },
                          "text_features": {
                              0: "num_text"
                          }
                      })

    probs = logits_per_image.cpu().numpy()
    print(logits_per_image, logits_per_text)

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]