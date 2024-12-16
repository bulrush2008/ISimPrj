
1, h5 file must be split into two files before training:
  - train.h5, and
  - test.h5
  24.12.10, 15:57

2, the field data may need to be normalize before training
  24.12.10, 16:03

3, the np.tofile() may may contain defects regarding the accuracy of the data
  24.12.11, 15:49

4, FNN 完成后，遇到了泛化的苦难：
  - 首先分析数据
  - 增加算例；
  - 调整网络层数
  - 增加后处理，实时输出对测试集的预测误差
  - 增加部署或存储功能
  24.12.16, 13:37