import cv2
import numpy as np
import torch
from torchvision.models import resnet50
from torchvision.transforms import ToTensor, Resize

def normalize_image(image):
    image = image - image.min()
    image = image / image.max()
    return image

def grad_cam(model, image):
    model.eval()

    # 预先定义预处理图像的方法
    transform = torch.nn.Sequential(
        Resize((224, 224)),
        ToTensor()
    )
    input_tensor = transform(image).unsqueeze(0)

    # 使用预训练的模型
    output = model(input_tensor)
    predicted_class = torch.argmax(output).item()

    # 计算梯度
    gradients = torch.autograd.grad(output[0, predicted_class], model.parameters(), retain_graph=True, create_graph=True)[0]

    # 获取目标特征图
    target_layer = model.layer4[-1]  # 使用ResNet-50的最后一个Layer4
    target_activation = target_layer.register_forward_hook(lambda self, input, output: input[0])

    # 计算权重
    weights = torch.mean(gradients, dim=(2, 3))

    # 计算Grad-CAM
    grad_cam = torch.zeros(target_activation.size()[2:])
    for i, weight in enumerate(weights):
        grad_cam += weight * target_activation[0, i, :, :]
    grad_cam = np.maximum(grad_cam.detach().numpy(), 0)

    # 归一化并生成热力图
    grad_cam = normalize_image(grad_cam)
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)

    # 叠加热力图
    output = cv2.resize(image, (224, 224))
    output = normalize_image(output)
    output = cv2.cvtColor(np.uint8(255 * output), cv2.COLOR_GRAY2RGB)
    output = cv2.addWeighted(output, 0.5, heatmap, 0.5, 0)

    target_activation.remove()  # 移除forward hook

    return output

# 加载预训练的ResNet-50模型
model = resnet50(pretrained=True)

# 虚拟创建一个大小为(4, 32, 32)的随机图像
image = np.random.rand(3, 32, 32)

# 显示Grad-CAM结果
output = grad_cam(model, image)
cv2.imshow("Grad-CAM", output)
cv2.waitKey(0)
cv2.destroyAllWindows()