
"""Train the model"""
import argparse

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation,
                                    RandomVerticalFlip, Resize, ToTensor)
from classifier import ImageClassifier
from utils import create_dir, get_dataset, validate

COMMON_TRANSFORM = Compose([
    Resize((224, 224), antialias=True),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = Compose([
    ToTensor(),
    COMMON_TRANSFORM
])

DATA_AUG_TRANSFORM = Compose([
    RandomCrop(300),
    RandomRotation(10),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
])

TRAIN_TRANSFORM = Compose([
    ToTensor(),
    DATA_AUG_TRANSFORM,
    COMMON_TRANSFORM
])




def train(train_loader: DataLoader,
          val_loader: DataLoader,
          model: ImageClassifier,
          optimizer,
          loss_func,
          num_epoch=10,
          gpu=True):
    """Train the model

    Args:
        train_loader (DataLoader): train dataset
        val_loader (DataLoader): validation dataset
        model (ImageClassifier): model to train
        optimizer (_type_): _description_
        loss_func (_type_): _description_
        num_classes (_type_): _description_
        num_epoch (int, optional): _description_. Defaults to 10.
        gpu (bool, optional): _description_. Defaults to True.
    """
    print("Start training ...")
    for epoch in range(num_epoch):
        mornitoring_loss = 0
        for batch, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data

            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs, training=True)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            mornitoring_loss += loss.item()
            if batch % 20 == 0:
                print("================================")
                print(f"Epoch {epoch}, at batch {batch}: {mornitoring_loss / 20}")
                print("Validation: ")
                model.eval()
                validate(val_loader, model, loss_func, gpu)
                model.train()
                mornitoring_loss = 0

def main():
    """Main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("data_directory")
    argument_parser.add_argument("--save_dir", default="save_dir")
    argument_parser.add_argument("--arch", default="resnet50")
    argument_parser.add_argument("--learning_rate", default=0.001, type=float)
    argument_parser.add_argument("--hidden_units", default= [512], nargs="+", type=int)
    argument_parser.add_argument("--epoch", default=10, type=int)
    argument_parser.add_argument("--gpu", action="store_true")
    argument_parser.add_argument("--batch", type=int, default=32)
    args = argument_parser.parse_args()


    class_to_idx, train_ds = get_dataset(args.data_directory + "/train",
                                         TRAIN_TRANSFORM, args.batch)
    
    print(class_to_idx)
    _, val_ds = get_dataset(args.data_directory + "/valid", VAL_TRANSFORM, 16)

    model = ImageClassifier(
        dropout_rate=.5,
        backbone=args.arch,
        hiddens=args.hidden_units,
        class_to_idx=class_to_idx
    )

    if args.gpu:
        model.cuda()

    train(train_ds, val_ds, model,
          Adam(model.parameters(), lr=args.learning_rate),
          CrossEntropyLoss(),
          args.epoch, args.gpu)

    create_dir(args.save_dir)
    ImageClassifier.save(model, args.save_dir + "/model.pth")

if __name__=="__main__":
    main()
