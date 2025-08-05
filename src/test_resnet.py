import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = urllib.request.urlopen(LABELS_URL).read().decode('utf-8').splitlines()

# Load and preprocess image
image_path = "/home/catkin_ws/src/amr_object_recognition/src/cat.jpg"  
img = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])
input_tensor = preprocess(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted_idx = torch.max(outputs, 1)

print(f"Predicted label: {labels[predicted_idx]}")
