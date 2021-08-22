from __future__ import division, print_function

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from tqdm import tqdm

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


class BaseModel:
    def __init__(self, model_name, num_classes, feature_extract=False, is_inception=False):
        self.is_inception = is_inception
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        self.model_name = model_name

    def __str__(self):
        return self.model_ft

    def __set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model_ft.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        self.model_ft = None
        # input_size = 0

        if model_name == "resnet50":
            """ Resnet50
            """
            self.model_ft = models.resnet50(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(feature_extract)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            self.model_ft = models.alexnet(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(feature_extract)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            # input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            self.model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(feature_extract)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            # input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            self.model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(feature_extract)
            self.model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model_ft.num_classes = num_classes
            # input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            self.model_ft = models.densenet121(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(feature_extract)
            num_ftrs = self.model_ft.classifier.in_features
            self.model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            # input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.model_ft = models.inception_v3(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(feature_extract)
            # Handle the auxilary net
            num_ftrs = self.model_ft.AuxLogits.fc.in_features
            self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

    def save_model(self, model, save_path, is_state_dict=False):
        if is_state_dict:
            model = model.state_dict()
        torch.save(model, save_path)

    def train_model(self, dataloaders, num_epochs=25):
        since = time.time()

        val_acc_history = []
        model = self.model_ft
        model = model.to(self.device)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                dl_iter = tqdm(dataloaders[phase], total=len(dataloaders[phase]))
                for inputs, labels in dl_iter:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if self.is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                self.save_model(model, save_path='results/{}/210716_{}.pth'.format(self.model_name, epoch))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.save_model(model.load_state_dict(best_model_wts),
                        save_path='results/{}/210716_best.pth'.format(self.model_name))
        return model, val_acc_history

    def predict(self,
                weights,
                inputs,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        model = torch.load(weights)
        model = model.to(device)
        inputs = inputs.to(device)
        with torch.no_grad():
            results = model(inputs)
        return results
