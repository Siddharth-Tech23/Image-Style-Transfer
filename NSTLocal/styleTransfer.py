import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # For efficient gradient descent.

from PIL import Image
import matplotlib.pyplot as plt

# To transform PIL Images to tensors.
import torchvision.transforms as transforms
# To train or load pretrained models.
from torchvision.models import vgg19, VGG19_Weights

# To deep copy the models.
import copy

# Setting the device to GPU (if available) in order to improve speed of training, generation etc.s
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# desired size of the output image
imsize = 512 if (device in ['cuda', 'mps']) else 128

# Give this an image and it resizes the image, converts it to a tensor.
loader = transforms.Compose([
         transforms.Resize(imsize),  # Resizing the image to our desired size.
         transforms.ToTensor()
    ])       # Converting PIL image to a tensor.


# PIL Images range from 0 to 255 but when transformed to tensors, their range becomes 0 to 1.
# Both the images need to be resized to have the same dimensions as each other.
# NN from tensor libraries are trained with values ranging from 0 to 1. 
# So, giving 0-255 images as input will not work as activated feature maps won't sense content and style.
# Helper function. If we give it the image name, it will load it.
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Loading the images
style_img = image_loader("./images/starry_night.jpeg")
content_img = image_loader("./images/sunset_skyline.jpg")

# If the dimensions of the style and content images are not same, it will throw an AssertionError.
# The backslash at the end is a line continuation character. It allows breaking a long line of code for better readability.
assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

# reconverting tensor to PIL image. 
unloader = transforms.ToPILImage()

# Set the plotting in interactive mode.
plt.ion()

# Helper function to show the tensor as a PIL image.
def imshow(tensor, title=None):
    image = tensor.cpu().clone()    # We clone the tensor to not do changes on it.
    image = image.squeeze(0)        # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)    # pause a bit so that plots are updated.

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

# Content Loss
class ContentLoss(nn.Module):

    def __init__(self, target):
        super().__init__()
        # We detach the target content from the tree used to dynamically compute the gradient
        # This is a stated value, not a variable. Otherwise the forward method of the criterion will throw an error.
        self.target = target.detach()   # We detach it so forward doesn't throw an error when we calculate loss.

    def forward(self, input):
        # self.target is the layer 4_2 of vgg
        # input is the image that we are iteratively changing.
        self.loss = F.mse_loss(input, self.target)  # calculating the mean square loss.
        return input

# Helper Function. This is for style loss.
def gram_matrix(input):
    a, b, c, d = input.size() # a = batch_size(=1)
    # b = number of feature maps
    # (c, d) = dimensions of a feature map (N=c*d)

    features = input.view(a * b, c * d) # resize F_XL to \hat F_XL

    G = torch.mm(features, features.t())    # Compute the gram product

    # We 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# Style Loss
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()  # Detach so forward doesn't throw error when calculating loss.

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Importing the model. Setting it to eval because some layers have different behaviour during training than evaluation.
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

# VGG networks are trained on images with each channel normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
# We will use them to normalize the image before sending it into the network. Used in original paper.
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# Helper class for the VGG.
# Create a module to normalize input image so that we can easily put it into NN.Sequential.
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # .view the mean and std to make them [Cx1x1] so that they can work
        # with image tensor of shape [BxCxHxW]. B-> Batch size.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # Normalize image
        return (img - self.mean) / self.std

# Desired depth layers to compute style/content losses:
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Helper function for the creation of a new, modified vgg network with custom layers.
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, we make a new Sequential to put it modules
    # that are supposed to be activated sequentially.
    model = nn.Sequential(normalization)

    # we're modifying the cnn to do whatever we want to do; tweaking it to generate images instead
    i=0 # increment every time we se a conv layer
    # iterate over everyting in the cnn model
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)  # Change it so our losses don't interfere with the network
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))  

        # adding the module to the model in sequential manner; recreating the model.
        model.add_module(name, layer)

        # If the layer is one of the content layers, we calculate the content loss and add it to the list.
        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()    # the image that we are generating
            content_loss = ContentLoss(target)
            # adding the loss to the modified vgg network
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # if the layer is one of the style layers of the model, we calculate the style loss and add it to the list.
        if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                # adding the loss to the modified vgg network
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses; removing layers responsible for classification.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# The starting image we are modifying and outputting. This is gonna be random noise at the start.
input_img = content_img.clone()
# input_img = torch.randn(content_img.data.size())

# Displaying the random image we begin with
plt.figure()
imshow(input_img, title='Input Image')

# This type of optimizer was preferred by the author of the paper.
def get_input_optimizer(input_img):
    # This line is to show that input is a parameter that requires a gradient.
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=400,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += (1/5)*sl.loss    # multiplied by 0.2 because it's in the paper.

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            # print stats every 50 epochs
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
