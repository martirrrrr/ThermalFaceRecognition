from train2 import CustomResNet
import torch
from torchsummary import summary
from torchvision import datasets, transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trasformations (as for train and test)
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# Load detaset
train_ds = datasets.ImageFolder('dataset_merged_split/train', transform=transform_test)

# Load model
model = CustomResNet(num_classes=len(train_ds.classes)).to(device)
model.load_state_dict(torch.load('best_thermal_model2.pth', map_location=device))

# Print summary
summary(model, input_size=(1, 112, 112))


from torchinfo import summary
summary(model, input_size=(32, 1, 112, 112))
