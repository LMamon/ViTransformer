import json, importlib.util, torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
import gradio as gr

REPO = "roylvzn/vit-cifar10"

#downlaod artifacts
weights = hf_hub_download(REPO, "pytorch_model.bin")
model = hf_hub_download(REPO, "vit_model.py")
classes = hf_hub_download(REPO, "cifar10_classes.json")

#import ViT
spec = importlib.util.spec_from_file_location("vit_model", model)
vm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)
ViTModel = vm.ViTModel

#build model
device = torch.device("cude" if torch.cuda.is_available() else "cpu")
vModel = ViTModel(num_classes=10).to(device).eval()
state = torch.load(weights, map_location=device)
vModel.load_state_dict(state)

with open(classes) as f:
    CLASSES = json.load(f)

#match the training image format
IMAGENET_MEAN=(0.485, 0.456, 0.406); IMAGENET_STD=(0.229, 0.224, 0.225)
preproc = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def predict(img: Image.Image):
    x = preproc(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(vModel(x), dim=1)[0].cpu().tolist()
    top3 = sorted(enumerate(probs), key=lambda t: t[1], reverse=True)[:3]
    return {CLASSES[i]:float(p) for i, p in top3}

gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil", label="Upload an image"),
             outputs=gr.Label(num_top_classes=3),
             title= "ViT on CIFAR10",
             description="""uploads an image and returns the top-3 classes from CIFAR10 \n
                            [airplane, automobile, bird, cat, deer, dog,
                                frog, horse, ship, truck]""").launch(share=False)

