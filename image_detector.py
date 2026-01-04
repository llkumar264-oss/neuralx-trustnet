from transformers import pipeline

# Hugging Face AI-image detector
image_detector = pipeline(
    "image-classification",
    model="umm-maybe/AI-image-detector"
)

def analyze_image(image_path):
    result = image_detector(image_path)[0]

    return {
        "image_fake_probability": round(result["score"], 3),
        "label": result["label"]
    }
