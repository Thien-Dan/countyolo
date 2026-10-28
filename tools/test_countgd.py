import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image
from engines.grounding_dino_counter import GroundingDinoCounter

def main():
    img_path = 'data/FSC147/images_384_VarV2/35.jpg'
    img = Image.open(img_path).convert('RGB')
    print(f"Testing CountGD-Box (Fine-tuned FSC-147) on image 35.jpg (GT: 115 tomatoes)")

    counter = GroundingDinoCounter(model_id="checkpoints/countgd_box_hf")
    for thr in [0.25, 0.20, 0.15, 0.10, 0.05]:
        res = counter.count_image(img, "tomatoes", box_threshold=thr, text_threshold=thr)
        print(f"CountGD-Box | Thr {thr:.2f} -> Đếm được: {res['count']} quả (Inference: {res['ms']:.1f} ms)")

if __name__ == "__main__":
    main()
