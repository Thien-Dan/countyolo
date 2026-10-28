import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    GroundingDinoForObjectDetection,
    GroundingDinoImageProcessor,
    GroundingDinoProcessor,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.convert_grounding_dino_to_hf import (
    get_grounding_dino_config,
    create_rename_keys,
    rename_key,
    read_in_q_k_v_encoder,
    read_in_q_k_v_text_enhancer,
    read_in_q_k_v_decoder,
)

def convert_countgd(ckpt_path="checkpoints/countgd_box.pth", output_dir="checkpoints/countgd_box_hf"):
    print(f"Loading CountGD-Box checkpoint from {ckpt_path}...")
    original = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    original_state_dict = {k.replace("module.", ""): v for k, v in original.items()}

    # CountGD-Box sử dụng backbone Swin-B (grounding-dino-base: embed_dim=128, depths=[2, 2, 18, 2])
    model_name = "grounding-dino-base"
    config = get_grounding_dino_config(model_name)

    print("Renaming and mapping keys to HuggingFace format...")
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(original_state_dict, config)

    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)
    read_in_q_k_v_encoder(new_state_dict, config)
    read_in_q_k_v_text_enhancer(new_state_dict, config)
    read_in_q_k_v_decoder(new_state_dict, config)

    print("Instantiating HuggingFace GroundingDinoForObjectDetection model...")
    model = GroundingDinoForObjectDetection(config)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {len(missing)} (should be only position_ids or expected ones)")
    print(f"Unexpected keys: {len(unexpected)} (e.g. exemplar head weights if any: {unexpected[:5]})")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving converted model and processor to {output_dir}...")
    model.save_pretrained(output_dir)

    image_processor = GroundingDinoImageProcessor()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    processor = GroundingDinoProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained(output_dir)

    print("Successfully converted CountGD-Box to HuggingFace format!")

if __name__ == "__main__":
    convert_countgd()
