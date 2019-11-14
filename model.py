import torch
import torch.nn as nn
import torch.nn.functional as F

n_classes = 20

class ConvUnit(nn.Module):
    def __init__(self, in_deg, out_deg, kernel=3):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_deg, out_deg, kernel_size=kernel)
        self.norm = nn.BatchNorm2d(num_features=out_deg)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.start_deg = 16
        deg = self.start_deg
        next_deg = self.start_deg * 2

        self.conv1 = ConvUnit(3, deg)
        self.conv2 = nn.Conv2d(deg, deg, kernel_size=3)
        self.conv3 = nn.Conv2d(deg, next_deg, kernel_size=3)

        deg = next_deg
        next_deg *= 2

        self.conv4 = nn.Conv2d(deg, next_deg, kernel_size=3)
        self.conv5 = nn.Conv2d(next_deg, next_deg, kernel_size=3)
        self.conv6 = nn.Conv2d(next_deg, next_deg, kernel_size=3)

        deg = next_deg
        next_deg *= 2

        self.conv7 = nn.Conv2d(deg, next_deg, kernel_size=3)
        self.conv8 = nn.Conv2d(next_deg, next_deg, kernel_size=3)
        self.conv9 = nn.Conv2d(next_deg, next_deg*2, kernel_size=3)

        self.fc1 = nn.Linear(self.start_deg*16, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.dropoutconv1 = nn.Dropout2d(0.05)
        self.dropoutconv2 = nn.Dropout2d(0.05)
        self.dropoutconv3 = nn.Dropout2d(0.05)

        self.dropoutdense1 = nn.Dropout(0.3)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.dropoutconv1(x)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.dropoutconv1(x)

        x = F.relu(self.conv4(x))
        x = self.dropoutconv2(x)
        x = F.relu(self.conv5(x))
        x = self.dropoutconv2(x)
        x = F.relu(F.max_pool2d(self.conv6(x), 3))
        x = self.dropoutconv2(x)

        x = F.relu(F.max_pool2d(self.conv7(x), 2))
        x = self.dropoutconv3(x)
        x = F.relu(self.conv8(x))
        x = self.dropoutconv3(x)
        x = F.relu(F.max_pool2d(self.conv9(x), 3))
        x = self.dropoutconv3(x)

        x = x.view(-1, self.start_deg*16)
        x = F.relu(self.fc1(x))
        x = self.dropoutdense1(x)
        return self.fc2(x)

class PartialVGG16(nn.Module):
    def __init__(self):
        super(PartialVGG16, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),

                                     nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),

                                     nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),

                                     nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True)
                                     )

    def forward(self, x):
        out = self.network(x)
        feature_map = out

        return feature_map



class PartialSSD300(nn.Module):
    """
    Encapsulates our partial VGG16 network and the prediction layers.
    """

    def __init__(self, n_classes):
        super(PartialSSD300, self).__init__()

        self.n_classes = n_classes

        self.base = PartialVGG16()
        self.pred_convs = PredictionLayers(n_classes)

        # Since lower level features have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in our feature map
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_centers = self.get_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: locations and class scores (i.e. w.r.t each prior box) for each image
        """

        feature_map = self.base(image)

        # Rescale the feature map with L2 norm
        norm = feature_map.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        feature_map = feature_map / norm  # (N, 512, 38, 38)
        feature_map = feature_map * self.rescale_factors  # (N, 512, 38, 38)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(feature_map)

        return locs, classes_scores



class PredictionLayers(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes with our partialVGG16 output.
    We predict the boxes as encoded offsets from the priors.
    The class scores represent the scores of each object class in each of the bounding boxes located.
    """

    def __init__(self, n_classes):
        super(PredictionLayers, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering for the feature map
        self.n_boxes = 4

        # Box prediction convolutions (predict offsets w.r.t prior-boxes)
        self.box_pred = nn.Conv2d(512, self.n_boxes * 4, kernel_size=3, padding=1)
        # Class prediction convolutions (predict classes in boxes)
        self.class_pred = nn.Conv2d(512, self.n_boxes * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_params()

    def init_params(self):
        # Taken from existing implementation
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, feature_map):
        """
        Forward propagation.

        :param feature_map: a tensor of dimensions (N, 512, 38, 38)
        :return: box and class scores for each image
        """
        batch_size = feature_map.size(0)

        # Predict boxes' bounds (as offsets w.r.t prior-boxes)
        pred_box = self.box_pred(feature_map)
        print(pred_box.shape)
        pred_box = pred_box.permute(0, 2, 3, 1).contiguous()
        pred_box = pred_box.view(batch_size, -1, 4)

        # Predict classes in boxes
        pred_class = self.class_pred(feature_map)
        pred_class = pred_class.permute(0, 2, 3, 1).contiguous()
        pred_class = pred_class.view(batch_size, -1, self.n_classes)

        return pred_box, pred_class
