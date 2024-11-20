#import modules
import torch

#module architecture and inference with image
from vit_model import ViTModel
from torchvision import transforms
from PIL import Image

# M
#define model and load weights
NUM_CLASSES = 10
model = ViTModel(NUM_CLASSES)
model.load_state_dict(torch.load('vit_cifar10.pth', map_location=torch.device('cpu')))
model.eval()

print("model loaded successfully")

image_path = '<image url>'
image = Image.open(image_path)

#standard transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = transform(image).unsqueeze(0)

#run inference
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    print(f'Predicted class: {predicted.item()}')