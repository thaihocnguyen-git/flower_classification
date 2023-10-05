"""Predict label from an image.
"""
import argparse
import json


import torch
import torch.nn.functional as F
from classifier import ImageClassifier
from utils import process_image, load_image

@torch.no_grad()
def predict(image_path, model, topk=5, gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    idx_to_class = {idx: class_label for class_label, idx in model.class_to_idx.items()}
    
    # TODO: Implement the code to predict the class from an image file
    image = load_image(image_path)
    image = process_image(image)
    image = image[None, :]
    with torch.no_grad():
        if gpu:
            image = image.cuda()
        logits = model(image)[0]
        logits = F.softmax(logits, dim=0)
        values, indices = torch.topk(logits, topk, dim=0)
        indices = [idx_to_class[idx.item()] for idx in indices]
        values = [i.item() for i in values]
        return values, indices


def main():
    """Main function
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("image_path", help="Path of image to predict.")
    args_parser.add_argument("checkpoint_path",
                             help="Path to model checkpoint.",
                             default="./model.pth"
                             )
    args_parser.add_argument("--top_k", type=int, default=1)
    args_parser.add_argument("--cat_to_names",
                             help="Path to name mapping file",
                             default="./cat_to_name.json")
    args_parser.add_argument("--gpu", action="store_true")

    args = args_parser.parse_args()

    print("[INFO] Load category dict...")
    with open(args.cat_to_names, encoding="utf-8") as reader:
        cat_to_name = json.load(reader)


    print("[INFO] Init model ...")
    model = ImageClassifier.from_path(args.checkpoint_path)
    confidences, indices = predict(args.image_path, model, topk=args.top_k, gpu=args.gpu)
    labels = [cat_to_name.get(class_label) for class_label in indices]
    print(f"""
Predictions:
    Labels: {labels},
    Confidence: {confidences})""")

if __name__ == "__main__":
    main()
