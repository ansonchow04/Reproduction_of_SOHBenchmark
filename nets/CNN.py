import torch
import torch.nn as nn


class ResBlock(nn.Module):
    # 输入通道数，输出通道数，步长
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            # 一维卷积，传入input_channel和output_channel，
            # 卷积核大小为3，即目标及其左右，stride步长，是1的话就是挨个读，padding表示填充，如果填充1、卷积核=3，则输出长度跟输入长度一致，注意区分长度和通道数
            nn.Conv1d(
                input_channel, output_channel, kernel_size=3, stride=stride, padding=1
            ),
            # 一维批量标准化，按通道独立进行，可以提高网络的稳定性和效率
            nn.BatchNorm1d(output_channel),
            # 逐个ReLU
            nn.ReLU(),
            # 再次卷积，在不改变特征数量和序列长度的前提下，进行特征的深入细化和重组合，stride锁为1保证不进行额外的下采样，保持长度
            nn.Conv1d(
                output_channel, output_channel, kernel_size=3, stride=1, padding=1
            ),
            # 再次一维批量标准化
            nn.BatchNorm1d(output_channel),
        )

        # 留空，表示了ResNet表达式y=f(x)+x中的+x项
        self.skip_connection = nn.Sequential()
        # 如果输入输出通道数不匹配，需要对x进行维度处理，这里用简单稳定的卷积来降维不太会导致梯度问题
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out += self.skip_connection(x)
        # 先完成特征的提取和融合，最后再通过 ReLU 引入非线性并作为下一层的输入
        out = self.relu(out)
        return out


class CNN(nn.Module):
    """
    输入形状:    (N, 4, 128)
                (批次大小, 特征通道数, 序列长度)
    """

    def __init__(self):
        super(CNN, self).__init__()
        # 输入形状：(N, 4, 128)，特征通道从4拓展到16，步长为1，因此序列长度保持128，输出形状：(N, 16, 128)
        self.layer1 = ResBlock(input_channel=4, output_channel=16, stride=1)    # N, 16, 128
        # 输入形状：(N, 16, 128)，特征通道从16拓展到32，步长为2，因此序列长度减半为64，输出形状：(N, 32, 64)
        self.layer2 = ResBlock(input_channel=16, output_channel=32, stride=2)   # N, 32, 64
        # 输入形状：(N, 32, 64)，特征通道从32拓展到64，步长为2，因此序列长度减半为32，输出形状：(N, 64, 32)
        self.layer3 = ResBlock(input_channel=32, output_channel=64, stride=2)   # N, 64, 32
        # 输入形状：(N, 64, 32)，特征通道从64拓展到96，步长为2，因此序列长度减半为16，输出形状：(N, 96, 16)
        self.layer4 = ResBlock(input_channel=64, output_channel=96, stride=2)   # N, 96, 16
        # 输入形状：(N, 96, 16)，特征通道从96拓展到128，步长为2，因此序列长度减半为8，输出形状：(N, 128, 8)
        self.layer5 = ResBlock(input_channel=96, output_channel=128, stride=2)  # N, 128, 8
        
        # 全连接层，128个通道，每个通道8个数据点，总共128*8个输入特征
        self.predictor = nn.Sequential(
            # 将 1024 维的输入特征向量，投影到一个较低维度的隐藏特征空间（128 维）
            nn.Linear(128 * 8, 128),
            # ReLU 激活函数，引入非线性
            nn.ReLU(),
            # 最终将隐藏特征映射到单一输出（1 维），适用于回归任务
            nn.Linear(128 ,1)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # out.size(0)：保留batch_size，而将剩余的维度展平成一个维度，即变成batchsize*n
        pred = self.predictor(out.view(out.size(0), -1))
        return pred

# 用于定义该文件的入口点
# 如果文件被直接运行，main.py 的 __name__ 变量被设置为 '__main__'，因此条件判断为 True
# 如果文件被import， __name__ 变量会被设置为它的模块名（例如 'CNN'）。因此条件判断为 False。
# 通常用来放置只有在直接运行该文件时才需要执行的代码
if __name__ == '__main__':
    x = torch.rand(30, 4, 128)
    
    net = CNN()
    y = net(x)
    
    print(x.shape, y.shape)

    num_params = sum(param.numel() for  param in net.parameters())
    print(num_params)