from transformers import TimesformerForVideoClassification, AutoImageProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(
    "facebook/timesformer-base-finetuned-k400"
)

model = TimesformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400"
).to(device)

model.eval()

