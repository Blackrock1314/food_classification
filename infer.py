import cv2
import mindspore as ms
import mindspore.nn as nn
from mindspore.numpy import argmax

ckpt_path = 'resnet50.ckpt'
data_path = '请补充路径'

# 处理测试图片
def normalize(image):
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)
    image = image / 1.0
    image = (image[:, :] - mean) / std
    image = image[:, :, ::-1].transpose((2, 0, 1))  
    return image

# 处理路径下所有图片
def pre_deal(data_path):
    image = cv2.imread(data_path)
    norm_img = normalize(image)
    images = [norm_img]
    images = ms.Tensor(images, ms.float32)
    return images

# 推理函数
def infer(ckpt_path, data_path):
    image = pre_deal(data_path)

    net = ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  10)
    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(net, param_dict)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = ms.Model(net, loss, metrics={"Accuracy": nn.Accuracy()})

    output = model.predict(image)
    pred = argmax(output, axis=1)
    return pred

infer(ckpt_path, data_path)