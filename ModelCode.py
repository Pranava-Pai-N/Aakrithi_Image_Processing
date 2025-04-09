import torch
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

MAX_SIZE = (1024, 1024)

def load_and_resize_image(image_path):
    img = Image.open(image_path)
    if img.size[0] * img.size[1] > 89478485:
        print("Image too large, resizing...")
        img.thumbnail(MAX_SIZE)
    return img


ayurvedic_prompts = [
    "Ayurvedic medicine",
    "Ayurvedic treatment center",
    "Panchakarma therapy",
    "Natural herbal remedies",
    "Traditional Indian healing",
    "Herbal powder and oils",
    "Ayurvedic massage",
    "Siddha or Ayurveda practice",
    "Holy basil and turmeric",
    "Bowl of dried medicinal herbs",
    "Ayurvedic detox therapy",
    "Spiritual yoga and healing",
    "Ashwagandha root and powder",
    "Boiling herbal concoction",
    "Ayurvedic spa setting",
    "Handmade herbal soap",
    "Mortar and pestle with herbs",
    "Ayurvedic doctor in traditional dress",
    "Ayurvedic clinic interior",
    "Holy plants in a healing ritual",
    "Neem and tulsi leaves",
    "Ayurveda-inspired diet plan",
    "Copper water vessel with lemon",
    "Meditation with Ayurvedic setup",
    "Natural treatment using roots and leaves"
]

non_ayurvedic_prompts = [
    "Modern medicine bottle",
    "City hospital room",
    "Technology device on desk",
    "Fast food on a tray",
    "Traffic on an urban road",
    "Surgeon in operation theater",
    "Laptop and phone on a table",
    "Chemical pills in a box",
    "Modern doctor's office",
    "Processed food items",
    "Doctor using a stethoscope",
    "CT scan or X-ray room",
    "Cityscape with traffic",
    "Corporate office interior",
    "Plastic packaged medication",
    "Digital health monitoring device",
    "Blood pressure monitor",
    "Busy pharmacy counter",
    "Industrial lab equipment",
    "Person typing on a keyboard",
    "Microscope in a lab",
    "Person eating pizza",
    "High-tech surgical tools",
    "Injection or syringe closeup",
    "Coffee mug with laptop setup"
]


all_prompts = ayurvedic_prompts + non_ayurvedic_prompts
labels = ["Ayurveda"] * len(ayurvedic_prompts) + ["Non-Ayurveda"] * len(non_ayurvedic_prompts)

ayurvedic_keywords = ["herbs", "oil", "ayurveda", "healing", "natural", "plant", "therapy"]

def predict_with_clip(image_path, confidence_threshold=0.60):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(all_prompts).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_inputs)
        logits_per_image, _ = clip_model(image, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    scores = {"Ayurveda": 0.0, "Non-Ayurveda": 0.0}
    for idx, prob in enumerate(probs):
        scores[labels[idx]] = max(scores[labels[idx]], prob)

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score >= confidence_threshold:
        return best_label, best_score, "CLIP"
    else:
        return None, best_score, "Low confidence"

def fallback_blip_classification(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(raw_image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

    caption_lower = caption.lower()
    for kw in ayurvedic_keywords:
        if kw in caption_lower:
            return "Ayurveda", caption, "BLIP fallback"

    return "Non-Ayurveda", caption, "BLIP fallback"

def classify_image(image_path):
    label, score, method = predict_with_clip(image_path)

    if method == "CLIP":
        return {
            "Type": str(label),
        }
    else:
        label, caption, method = fallback_blip_classification(image_path)
        return {
            "Type": str(label),
        }
  