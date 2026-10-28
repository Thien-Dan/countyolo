import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image
from engines.grounding_dino_counter import GroundingDinoCounter

def main():
    img_path = 'data/FSC147/images_384_VarV2/35.jpg'
    img = Image.open(img_path).convert('RGB')
    print(f"Image size: {img.size}")

    counter = GroundingDinoCounter()
    for prompt in ["tomato.", "a tomato.", "tomatoes.", "small red tomato."]:
        for thr in [0.15, 0.10, 0.05, 0.02]:
            res = counter.count_image(img, prompt, box_threshold=thr, text_threshold=thr)
            print(f"Prompt '{prompt}' | Thr {thr:.2f} -> Count: {res['count']}")

if __name__ == "__main__":
    main()
