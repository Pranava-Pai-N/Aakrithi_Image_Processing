import torch
import clip
from PIL import Image
import gc
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract

device = "cpu"

clip_model = None
clip_preprocess = None
blip_processor = None
blip_model = None

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
    "Natural treatment using roots and leaves",
    "Herbal tea with cumin and ginger",
    "Ayurvedic facial therapy",
    "Healing stones with herbs",
    "Triphala powder and fruit",
    "Organic herbal tinctures",
    "Traditional ayurvedic scriptures",
    "Dry herbs in glass jars",
    "Oil pulling with sesame oil",
    "Chyawanprash herbal jam",
    "Ayurvedic nutrition consultation",
    "Cumin water detox",
    "Shatavari plant remedy",
    "Ginger and turmeric decoction",
    "Ayurveda in daily routine",
    "Herbal smoke cleansing",
    "Medicinal herb garden",
    "Ayurveda lifestyle book",
    "Ayurvedic pulse diagnosis",
    "Bottle of Ayurvedic syrup",
    "Marma therapy session",
    "Amla powder for hair and skin",
    "Natural immunity boosters",
    "Ayurvedic golden milk",
    "Cleansing herbal juices",
    "Ayurvedic retreat",
    "Ayurvedic foot bath",
    "Mulethi sticks for cough",
    "Herbal incense in meditation",
    "Traditional Kerala Ayurveda",
    "Kapha balancing diet",
    "Herbal body wrap",
    "Sandalwood paste application",
    "Turmeric face mask",
    "Ayurvedic diet for digestion",
    "Herbal ghee preparation",
    "Ayurvedic healing altar",
    "Handmade Ayurvedic candles",
    "Herbal steam therapy",
    "Ayurvedic tonics in bottles",
    "Ayurveda consultation with practitioner",
    "Dhoop sticks and herbs",
    "Ayurveda dosha chart",
    "Coriander and fennel tea",
    "Ayurvedic herb grinder",
    "Herbal supplement capsules",
    "Boiling neem leaves",
    "Yoga and Ayurveda combo session",
    "Sun-dried medicinal roots",
    "Ayurvedic skin care set",
    "Copper tongue scraper",
    "Cumin and ajwain detox water",
    "Handwritten Ayurvedic prescription",
    "Ayurvedic remedy instructions",
    "Doctor's note with herbal treatments",
    "Sanskrit Ayurvedic script",
    "Page from Ayurveda book",
    "Text describing herbal decoctions",
    "Prescription with cumin and ginger"
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
    "Coffee mug with laptop setup",
    "Capsules in a transparent strip",
    "Surgical gloves and mask",
    "Heart rate monitor screen",
    "Smartwatch fitness tracking",
    "Hospital bed with equipment",
    "Fast food burger meal",
    "IV drip setup",
    "Modern health supplement packaging",
    "White coat doctor with clipboard",
    "MRI machine in clinic",
    "Tech gadgets with health app",
    "Pharmaceutical advertisement",
    "Soda can with fries",
    "Crowded emergency room",
    "Modern pill organizer",
    "Fitness tracker close-up",
    "City skyline with pollution",
    "Desktop computer with reports",
    "Plastic bottles of vitamins",
    "Sterile lab setting",
    "Box of cough syrup",
    "Office desk with documents",
    "Ambulance in traffic",
    "Processed snack packs",
    "Online health consultation",
    "Medical chart on tablet",
    "Pills spilled on table",
    "Surgical face mask in hand",
    "Bottled energy drink",
    "X-ray images on display",
    "Blood sample in vial",
    "Digital thermometer reading",
    "Fitness coach with gadgets",
    "Disposable syringe packet",
    "Artificial sweeteners",
    "PowerPoint health presentation",
    "Surgery with robotic arms",
    "Skin care with chemical serum",
    "Health app interface",
    "Plastic pill blister pack",
    "Busy tech-driven clinic",
    "Athlete drinking sports drink",
    "Medical AI device demo",
    "Pharmacy shelf with labels",
    "High sugar soda bottle",
    "Biotech company office",
    "Keyboard with health analytics",
    "Doctor on a video call",
    "Smartphone showing medicine app",
    "Virtual health dashboard",
    "Injection for flu shot"
]



all_prompts = ayurvedic_prompts + non_ayurvedic_prompts
labels = ["Ayurveda"] * len(ayurvedic_prompts) + ["Non-Ayurveda"] * len(non_ayurvedic_prompts)

ayurvedic_keywords = ["herbs", "oil", "ayurveda", "healing", "natural", "plant", "therapy"]

def predict_with_clip(image_path, confidence_threshold=0.60):
    global clip_model, clip_preprocess
    if clip_model is None or clip_preprocess is None:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(all_prompts).to(device)

    with torch.no_grad():
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
    global blip_model, blip_processor
    if blip_model is None or blip_processor is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

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

def ocr_classification(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    text_lower = text.lower()

    for kw in ayurvedic_keywords:
        if kw in text_lower:
            return "Ayurveda"
    return "Non-Ayurveda"


def classify_image(image_path):
    label = ocr_classification(image_path)

    if label == "Ayurveda":
        result = {
            "Type": label,
        }
    else:
        label, score, method = predict_with_clip(image_path)
        if method == "CLIP" and label is not None:
            result = {
                "Type": label,
            }
        else:
            label, caption, method = fallback_blip_classification(image_path)
            result = {
                "Type": label,
            }

    torch.cuda.empty_cache()
    gc.collect()
    return result