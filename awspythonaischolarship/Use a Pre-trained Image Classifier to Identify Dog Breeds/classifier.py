
import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import version

# Load pre-trained models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

# Dictionary to hold the models for easy access
models = {
    'resnet': resnet18,
    'alexnet': alexnet,
    'vgg': vgg16
}

# Load ImageNet labels from a text file
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def classifier(img_path, model_name):
    # Load the image from the specified path
    img_pil = Image.open(img_path)

    # Define the preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),          # Resize the image to 256 pixels
        transforms.CenterCrop(224),     # Crop the center 224x224 pixels
        transforms.ToTensor(),           # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    # Preprocess the image
    img_tensor = preprocess(img_pil)

    # Add a batch dimension to the tensor
    img_tensor.unsqueeze_(0)

    # Check PyTorch version to determine how to handle the tensor
    pytorch_ver = version.split('.')

    # For PyTorch versions 0.4 and higher, we no longer need to wrap the tensor in a Variable
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)  # Set requires_grad to False for inference
    else:
        # For versions less than 0.4, wrap the tensor in a Variable
        data = Variable(img_tensor, volatile=True)

    # Select the model based on the provided model name
    model = models[model_name]

    # Set the model to evaluation mode (disables dropout, batch normalization, etc.)
    model.eval()

    # Apply the model to the input tensor
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        output = model(img_tensor)  # Directly apply the model to the tensor
    else:
        output = model(data)  # Apply the model to the Variable

    # Return the index corresponding to the predicted class
    pred_idx = output.data.numpy().argmax()
    return imagenet_classes_dict[pred_idx]  # Return the human-readable label for the predicted class