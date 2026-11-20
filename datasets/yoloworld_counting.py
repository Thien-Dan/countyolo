"""FSC147 targets for point and partial-box detector training."""
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from datasets.fsc147_dataset import load_class_names
from models.yoloworld_counting import exemplar_point_indices

# Author FSC147-D singularized & corrected prompts (from CountGD/CountVid train/val/test label maps).
FSC147D_CLASS_MAP = {
    'alcohol bottles': 'alcohol bottle', 'ants': 'ant', 'apples': 'apple', 'baguette rolls': 'baguette roll',
    'balls': 'ball', 'bananas': 'banana', 'beads': 'bead', 'bees': 'bee', 'birds': 'bird',
    'birthday candles': 'birthday candle', 'biscuits': 'biscuit', 'boats': 'boat', 'books': 'book',
    'bottle caps': 'bottle cap', 'bottles': 'bottle', 'bowls': 'bowl', 'boxes': 'box',
    'bread rolls': 'bread roll', 'bricks': 'brick', 'buffaloes': 'buffalo', 'bullets': 'bullet',
    'buns': 'bun', 'calamari rings': 'calamari ring', 'camels': 'camel', 'candles': 'candle',
    'candy pieces': 'candy piece', 'cans': 'can', 'caps': 'cap', 'carrom board pieces': 'carrom board piece',
    'cars': 'car', 'cartridges': 'cartridge', 'cashew nuts': 'cashew nut', 'cassettes': 'cassette',
    'cement bags': 'cement bag', 'cereals': 'cereal', 'chairs': 'chair',
    'chewing gum pieces': 'chewing gum piece', 'chicken wings': 'chicken wing', 'clams': 'clam',
    'coffee beans': 'coffee bean', 'coins': 'coin', 'comic books': 'comic book',
    'cotton balls': 'cotton ball', 'cows': 'cow', 'crab cakes': 'crab cake', 'cranes': 'crane',
    'crayons': 'crayon', 'croissants': 'croissant', 'crows': 'crow', 'cupcake tray': 'cupcake',
    'cupcakes': 'cupcake', 'cups': 'cup', 'deers': 'deer', 'donuts tray': 'donut', 'eggs': 'egg',
    'elephants': 'elephant', 'finger foods': 'finger food', 'fishes': 'fish', 'flamingos': 'flamingo',
    'flower pots': 'flower pot', 'flowers': 'flower', 'fresh cut': 'package of fresh cut fruit',
    'geese': 'goose', 'gemstones': 'gemstone', 'go game': 'go game piece', 'goats': 'goat',
    'grapes': 'grape', 'green peas': 'green pea', 'horses': 'horse',
    'hot air balloons': 'hot air balloon', 'instant noodles': 'instant noodle',
    'jade stones': 'jade stone', 'keyboard keys': 'keyboard key', 'kidney beans': 'kidney bean',
    'kitchen towels': 'kitchen towel', 'kiwis': 'kiwi', 'legos': 'lego', 'lighters': 'lighter',
    'm&m pieces': 'm&m piece', 'macarons': 'macaron', 'marbles': 'marble', 'markers': 'marker',
    'matches': 'match', 'meat skewers': 'meat skewer', 'milk cartons': 'milk carton',
    'mini blinds': 'mini blind', 'mosaic tiles': 'mosaic tile', 'nails': 'nail', 'nuts': 'nut',
    'onion rings': 'onion ring', 'oranges': 'orange', 'oyster shells': 'oyster shell',
    'oysters': 'oyster', 'peaches': 'peach', 'pearls': 'pearl', 'pencils': 'pencil',
    'penguins': 'penguin', 'pens': 'pen', 'people': 'person', 'peppers': 'pepper',
    'pigeons': 'pigeon', 'pills': 'pill', 'plates': 'plate', 'polka dot tiles': 'polka dot tile',
    'polka dots': 'polka dot', 'potato chips': 'potato chip', 'potatoes': 'potato',
    'prawn crackers': 'prawn cracker', 'red beans': 'red bean', 'rice bags': 'rice bag',
    'roof tiles': 'roof tile', 'sauce bottles': 'sauce bottle', 'sausages': 'sausage',
    'screws': 'screw', 'sea shells': 'sea shell', 'seagulls': 'seagull', 'shallots': 'shallot',
    'shirts': 'shirt', 'shoes': 'shoe', 'skis': 'ski', 'spring rolls': 'spring roll',
    'stairs': 'stair', 'stamps': 'stamp', 'stapler pins': 'stapler pin',
    'sticky notes': 'sticky note', 'strawberries': 'strawberry', 'straws': 'straw',
    'swans': 'swan', 'toilet paper rolls': 'toilet paper roll', 'tomatoes': 'tomato',
    'tree logs': 'tree log', 'watches': 'watch', 'windows': 'window', 'zebras': 'zebra',
}
FSC147D_IMAGE_OVERRIDES = {
    **{f'{i}.jpg': 'cupcake holder' for i in [
        4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239, 4240, 4243, 4244, 4245, 4246, 4247, 4248,
        4249, 4251, 4252, 4253, 4255, 4256, 4258, 4259, 4261, 4262, 4263, 4264, 4265, 4266, 4267,
        4269, 4270, 4271, 4272, 4273, 4275, 4276, 4281, 4282, 5446, 5456, 5777, 5780, 5782, 5784,
        5786, 5787, 5789, 5790, 6001, 6002, 6003, 6004, 6011, 6015, 6016, 6020, 6022, 6023, 6026,
        6027, 6032]},
    '3696.jpg': 'round dessert', '3697.jpg': 'round dessert',
    '5449.jpg': 'ice cream cone', '5664.jpg': 'donut holder', '7611.jpg': 'yellow lego stud',
}


def load_fsc147d_class_names(root: str) -> dict:
    raw = load_class_names(root)
    return {name: FSC147D_IMAGE_OVERRIDES.get(name, FSC147D_CLASS_MAP.get(cls, cls))
            for name, cls in raw.items()}


def _resize_to_short(w, h, short, max_size=1333):
    if max_size is not None:
        min_orig, max_orig = float(min(w, h)), float(max(w, h))
        if max_orig / min_orig * short > max_size:
            short = int(round(max_size * min_orig / max_orig))
    if (w <= h and w == short) or (h <= w and h == short):
        return w, h
    if w < h:
        return short, int(short * h / w)
    return int(short * w / h), short


def _crop_sample(image, points, supervision_points, boxes, pre_short, crop_min=384, crop_max=600):
    w, h = image.size
    nw, nh = _resize_to_short(w, h, pre_short)
    image = image.resize((nw, nh), Image.Resampling.BILINEAR)
    sx, sy = nw / float(w), nh / float(h)
    points = points * [sx, sy]
    supervision_points = supervision_points * [sx, sy]
    if len(boxes):
        boxes = boxes * np.array([sx, sy, sx, sy], dtype=np.float32)
    cw = random.randint(min(crop_min, nw), min(nw, crop_max))
    ch = random.randint(min(crop_min, nh), min(nh, crop_max))
    top = random.randint(0, nh - ch)
    left = random.randint(0, nw - cw)
    image = image.crop((left, top, left + cw, top + ch))
    # Keep targets whose 2x2 point box remains inside the crop (author transforms.crop).
    for arr_name, pts in [('points', points), ('supervision', supervision_points)]:
        pass
    p_boxes = np.stack([points[:, 0] - 1, points[:, 1] - 1, points[:, 0] + 1, points[:, 1] + 1], axis=-1) if len(points) else np.empty((0, 4), dtype=np.float32)
    s_boxes = np.stack([supervision_points[:, 0] - 1, supervision_points[:, 1] - 1, supervision_points[:, 0] + 1, supervision_points[:, 1] + 1], axis=-1) if len(supervision_points) else np.empty((0, 4), dtype=np.float32)
    if len(s_boxes):
        s_boxes = s_boxes - np.array([left, top, left, top], dtype=np.float32)
        s_boxes[:, 0::2] = np.clip(s_boxes[:, 0::2], 0, cw)
        s_boxes[:, 1::2] = np.clip(s_boxes[:, 1::2], 0, ch)
        keep = (s_boxes[:, 2] > s_boxes[:, 0]) & (s_boxes[:, 3] > s_boxes[:, 1])
        supervision_points = (s_boxes[keep, :2] + s_boxes[keep, 2:]) / 2.0
        p_boxes = p_boxes - np.array([left, top, left, top], dtype=np.float32)
        p_boxes[:, 0::2] = np.clip(p_boxes[:, 0::2], 0, cw)
        p_boxes[:, 1::2] = np.clip(p_boxes[:, 1::2], 0, ch)
        points = (p_boxes[keep, :2] + p_boxes[keep, 2:]) / 2.0
    if len(boxes):
        boxes = boxes - np.array([left, top, left, top], dtype=np.float32)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, cw)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, ch)
        keep_ex = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep_ex]
    return image, points, supervision_points, boxes


def prepare(image, record, short=640, flip=False, crop=False):
    points = np.asarray(record['points'], dtype=np.float32).reshape(-1, 2).copy()
    boxes = np.asarray([np.concatenate((np.min(b,axis=0),np.max(b,axis=0)))
                        for b in record.get('box_examples_coordinates', [])],dtype=np.float32).reshape(-1,4)
    raw_exemplar_count = len(boxes)
    supervision_points=points.copy()
    # Original odvg.py updates exemplar-associated targets before transforms.
    indices,matched_boxes=exemplar_point_indices(points,boxes)
    for idx,box in zip(indices,matched_boxes):
        supervision_points[idx]=(box[:2]+box[2:])/2
    w, h = image.size
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        points[:, 0] = w-points[:, 0]
        supervision_points[:,0]=w-supervision_points[:,0]
        if len(boxes):
            boxes[:, [0,2]] = w-boxes[:, [2,0]]
    if crop:
        image, points, supervision_points, boxes = _crop_sample(
            image, points, supervision_points, boxes, random.choice([400, 500, 600]))
        w, h = image.size
    ratio = min(short/min(w,h), 1333/max(w,h))
    nw, nh = round(w*ratio), round(h*ratio)
    image = image.resize((nw,nh), Image.Resampling.BILINEAR)
    # Top-left padding to stride32. Targets are normalized by padded dimensions.
    pw, ph = (nw+31)//32*32, (nh+31)//32*32
    canvas = np.full((ph,pw,3), 114, dtype=np.uint8)
    canvas[:nh,:nw] = np.asarray(image)
    if len(points):
        points *= [nw/w,nh/h]
        supervision_points *= [nw/w,nh/h]
        points /= [pw,ph]
        supervision_points /= [pw,ph]
    if len(boxes):
        boxes *= np.array([nw/w,nh/h]*2, dtype=np.float32)
        boxes /= np.array([pw,ph]*2, dtype=np.float32)
    exemplar_indices, exemplar_boxes = exemplar_point_indices(supervision_points, boxes)
    return dict(image=torch.from_numpy(canvas.copy()).permute(2,0,1).float()/255,
        points=torch.from_numpy(points), exemplar_indices=torch.from_numpy(exemplar_indices),
        supervision_points=torch.from_numpy(supervision_points),
        exemplar_boxes=torch.from_numpy(exemplar_boxes), original_size=(w,h),
        resized_size=(nw,nh), padded_size=(pw,ph), raw_exemplar_count=raw_exemplar_count)


class CountingData:
    def __init__(self, root, split, exclude=(), use_fsc147d_prompts=True, random_crop=False):
        self.root = Path(root)
        self.annotations = json.loads((self.root/'annotation_FSC147_384.json').read_text())
        self.names = [n for n in json.loads((self.root/'Train_Test_Val_FSC_147.json').read_text())[split] if n not in set(exclude)]
        self.classes = load_fsc147d_class_names(str(self.root)) if use_fsc147d_prompts else load_class_names(str(self.root))
        self.split = split
        self.random_crop = random_crop

    def get(self, name, augment=False):
        image = Image.open(self.root/'images_384_VarV2'/name).convert('RGB')
        record=self.annotations[name]
        if name=='7171.jpg':
            record=dict(record,box_examples_coordinates=[])
        sample = prepare(image, record,
            short=random.choice([480,512,544,576,608,640,672,704,736,768,800]) if augment else 800,
            flip=augment and random.random()<.5,
            crop=augment and self.random_crop and random.random()<.5)
        sample.update(name=name, prompt=self.classes[name])
        return sample
