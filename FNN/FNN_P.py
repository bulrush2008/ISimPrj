
import torch
import torch.nn as nn
import pandas

class Classfier(nn.Module):
  # 初始化 PyTorch 父类
  def __init__(self):
    super().__init__()

    self.model = nn.Sequential(
        nn.Linear(784,200),
        nn.Sigmoid(),
        nn.LayerNorm(200),
        nn.Linear(200,10),
        nn.Sigmoid()
    )

    self.loss_function = nn.MSELoss()

    self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

    self.counter = 0
    self.progress = []
    pass

  # 前向传播
  def forward(self, inputs):
    return self.model(inputs)

  def train(self, inputs, targets):
    outputs = self.forward(inputs)
    # 计算损失值
    loss = self.loss_function(outputs, targets)

    self.counter += 1
    if(self.counter%10 == 0):
      self.progress.append(loss.item()) # loss 是什么类型？tensor-scalar
      pass

    if(self.counter%10000 == 0):
      print(self.counter)
      pass

    # 梯度归零，反向传播，更新学习参数
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def plot_progress(self):
    df = pandas.DataFrame(self.progress, columns=['loss'])
    df.plot(ylim=(0.0,1.0), figsize=(16.,8.), alpha=0.1, marker='.', grid=True, yticks=(0.,0.25,0.5))
    pass
  pass

from torch.utils.data import Dataset
import pandas
import torch
import matplotlib.pyplot as plt

class MnistDataset(Dataset):
  def __init__(self, csv_file):
    # csv_file: absolute address of the data file
    self.data_df = pandas.read_csv(csv_file, header=None)
    pass

  def __len__(self):
    return len(self.data_df)

  def __getitem__(self, index):
    # 目标函数（标签）
    label = self.data_df.iloc[index,0]
    target = torch.zeros((10))
    target[label] = 1.0 # one-hot representation

    # 图像数据，取值范围[0-255]，标准化为[0-1]
    image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values) / 255.0

    # 返回标签、图像数据张量，以及目标张量
    return label, image_values,  target

  def plot_image(self, index):
    img = self.data_df.iloc[index,1:].values.reshape(28,28)
    plt.title("label = "+str(self.data_df.iloc[index,0]))
    plt.imshow(img, interpolation='none',cmap='Blues')
    pass

  pass

if __name__=="__main__":
  #mnist_dataset = MnistDataset('mount/My Drive/Colab Notebooks/mnist_data/mnist_train.csv')
  #mnist_dataset.plot_image(9)

  #data = mnist_dataset(100) # not callable
  #data = mnist_dataset[9]
  #print(data[0])
  pass

# 创建神经网络
C = Classfier()

from pathlib import Path
dirMNistTrain = Path('../FSCases/MNistData/mnist_train.csv')

# read the train data
mnist_dataset = MnistDataset(dirMNistTrain)

# 在 MNIST 数据集，训练神经网络
epochs = 1

for i in range(epochs):
  print('training epoch', i+1, "of", epochs)

  for label, image_data_tensor, target_tensor in mnist_dataset:
    C.train(image_data_tensor, target_tensor)
    pass
  pass

# 绘制损失函数历史曲线
C.plot_progress()

# 对结果的测试
dirMNistTest = Path('../FSCases/MNistData/mnist_test.csv')
mnist_test_dataset = MnistDataset(dirMNistTest)

# 挑一幅图像
record = 19

mnist_test_dataset.plot_image(record)
image_data = mnist_test_dataset[record][1]

# 调用训练后的神经网络模型
output = C.forward(image_data)

# 绘制输出张量
pandas.DataFrame(output.detach().numpy()).plot(kind='bar',legend=False, ylim=(0,1))

# 测试神经网络模型

score = 0.0
items = 0.0

for label, image_data_tensor, target_tensor in mnist_test_dataset:
  answer = C.forward(image_data_tensor).detach().numpy()

  if(answer.argmax() == label):
    score += 1
    pass
  items += 1
  pass

print(score, items, score/items)