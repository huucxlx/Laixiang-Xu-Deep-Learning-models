from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#读取图片
img = Image.open("img/test.jpg")
#将图片转换为torch tensor
img_tensor = transforms.ToTensor()(img)

#定义平移变换矩阵
#0.1表示将图片向左平移图片宽的百分比
#0.2表示将图片向上平移图片高的百分比
theta = torch.tensor([[1,0,0.1],[0,1,0.2]],
                     dtype=torch.float)
#根据变换矩阵来计算变换后图片的对应位置
grid = F.affine_grid(theta.unsqueeze(0),
               img_tensor.unsqueeze(0).size(),align_corners=True)
#默认使用双向性插值，可以通过mode参数设置
output = F.grid_sample(img_tensor.unsqueeze(0),
			   grid,align_corners=True)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.array(img))
plt.title("original image")

plt.subplot(1,2,2)
plt.imshow(output[0].numpy().transpose(1,2,0))
plt.title("stn transform image")

plt.show()
