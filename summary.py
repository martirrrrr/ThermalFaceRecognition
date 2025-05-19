from train2 import CustomResNet
import torch
from torchsummary import summary
from torchvision import datasets, transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trasformazioni (devono essere uguali a quelle usate nel training)
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# Carica dataset solo per ottenere il numero di classi
train_ds = datasets.ImageFolder('dataset_merged_split/train', transform=transform_test)

# Istanzia e carica modello
model = CustomResNet(num_classes=len(train_ds.classes)).to(device)
model.load_state_dict(torch.load('best_thermal_model2.pth', map_location=device))

# Stampa riepilogo
summary(model, input_size=(1, 112, 112))


from torchinfo import summary
summary(model, input_size=(32, 1, 112, 112))  # batch_size=32 opzionale