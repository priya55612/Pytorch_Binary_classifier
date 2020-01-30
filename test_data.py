import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse




#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")


model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=False, num_classes=2)
model.load_state_dict(torch.load('/home/priyanka/Desktop/fingerprint_model.pth', map_location='cpu'))

model.eval()
print(model)

test_data_dir = '/home/priyanka/Desktop/Test_data_distal_non_distal/'
test_transforms = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])
data = datasets.ImageFolder(test_data_dir, transform=test_transforms)
classes = data.classes

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num):
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data,sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels
to_pil = transforms.ToPILImage()
images, labels = get_random_images(20)
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    #sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    plt.title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig('/home/priyanka/Desktop/results/image_'+str(ii)+'.png')
           
