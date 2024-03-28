from model import ResNet, block


def ResNet18(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


# we have two part model, encoder and decoder

# loss1 feature loss?
# loss2 reconstruction loss?
# loss3 VAE loss?

# for every input is height 480, weight 640, feature map is 20*15 would be super big?
