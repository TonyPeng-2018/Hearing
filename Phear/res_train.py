from model import ResNet, ResNet18Dec
from dataset import load_data
from torchinfo import summary # visiualize the model

encoder = ResNet(layers=[2,2,2,2], img_channels=1)
decoder = ResNet18Dec( layers=[2,2,2,2], out_channel=1)

summary(encoder, (64, 1, 640, 480)) # batch size, channel, height, width

summary(decoder, (64, 512, 20, 15)) # batch size, channel, height, width
# we have two part model, encoder and decoder

# loss1 feature loss?
# loss2 reconstruction loss?
# loss3 VAE loss?

# for every input is height 480, weight 640, feature map is 20*15 would be super big?
