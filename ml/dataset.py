# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class MonDataset(Dataset):
    def __init__(self, dossier, transform=None):
        self.dossier   = dossier
        self.transform = transform
        self.classes   = sorted(os.listdir(dossier))
        self.images    = []
        self.labels    = []
        for idx, classe in enumerate(self.classes):
            classe_dir = os.path.join(dossier, classe)
            if os.path.isdir(classe_dir):
                for f in os.listdir(classe_dir):
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(classe_dir, f))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_dataloaders(batch_size=32):
    # Augmentation pour l'entrainement
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Pas d'augmentation pour val/test
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dir = './archive/seg_train/seg_train'
    test_dir  = './archive/seg_test/seg_test'

    train_full   = MonDataset(train_dir, transform=transform_train)
    test_dataset = MonDataset(test_dir,  transform=transform_test)

    val_size   = int(0.15 * len(train_full))
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"Classes : {train_full.classes}")
    print(f"Train   : {len(train_dataset)} | Val : {len(val_dataset)} | Test : {len(test_dataset)}")

    return train_loader, val_loader, test_loader
