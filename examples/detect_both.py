import os
from nemotron_detector import NemotronDetector


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = r"sample.jpg"
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

word_detector = NemotronDetector.from_variant("word")
word_result = word_detector.process(
    image_path=IMAGE_PATH,
    output_dir=OUTPUT_DIR,
    save_labeled=True,
    save_mask=True,
    save_json=True,
    save_paired=True,
)

line_detector = NemotronDetector.from_variant("line")
line_result = line_detector.process(
    image_path=IMAGE_PATH,
    output_dir=OUTPUT_DIR,
    save_labeled=True,
    save_mask=True,
    save_json=True,
    save_paired=True,
)

print("Word detections:", len(word_result["detections"]))
print("Line detections:", len(line_result["detections"]))
print("Word outputs:", word_result["exports"])
print("Line outputs:", line_result["exports"])
