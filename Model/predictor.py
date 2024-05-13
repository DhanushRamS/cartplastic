import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

classes = ['battery','biological','cardboard','clothes','glass','metal','paper','plastic','shoes','trash']


transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
model = torch.jit.load('./model2.pt', map_location=torch.device('cpu'))


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    prob, preds  = torch.max(yb, dim=1)
    return classes[preds[0].item()]

def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))
    example_image = transformations(image)
    print("The image resembles", predict_image(example_image, model) + ".")

predict_external_image('plastic.jpg')