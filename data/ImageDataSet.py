import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageDataSet(Dataset):
    def __init__(self, data, data_shape=(200, 200, 3), is_transform=False):
        super().__init__()
        self.data = data
        self.image_width = data_shape[0]
        self.image_height = data_shape[1]
        self.image_channel = data_shape[2]
        self.is_transform = is_transform

        if self.is_transform:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_width, self.image_height)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((self.image_width, self.image_height), scale=(0.8, 1.0), ratio=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_width, self.image_height)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index].file_path)
        input_tensor = self.transform(image)
        label = int(self.data[index].age)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return input_tensor, label_tensor
