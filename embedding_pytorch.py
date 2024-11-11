import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform  #用来初始化网络权重
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Training Data:
#   Troll2 is great!
#   Gymkata is great!


class WordEmbeddingFromScrath(L.LightningModule):
    def __init__(self):
        super().__init__()

        min_value = -0.5
        max_value = 0.5

        self.input1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.output1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.loss = nn.CrossEntropyLoss()

    def forward(self,input):  #input 是 一个包含其中一个输入token的 one-hot 编码 eg:[[1.,0.,0.,0.]]
        input = input[0]  #去掉一个括号 [1.,0.,0.,0.]

        input_to_top_hidden = ((input[0]*self.input1_w1)+
                               (input[1]*self.input2_w1)+
                               (input[2]*self.input3_w1)+
                               (input[3]*self.input4_w1))

        input_to_bottom_hidden = ((input[0]*self.input1_w2)+
                                  (input[1]*self.input2_w2)+
                                  (input[2]*self.input3_w2)+
                                  (input[3]*self.input4_w2))

        output1 = ((input_to_top_hidden * self.output1_w1)+
                   (input_to_bottom_hidden * self.output1_w2))
        output2 = ((input_to_top_hidden * self.output2_w1)+
                   (input_to_bottom_hidden * self.output2_w2))
        output3 = ((input_to_top_hidden * self.output3_w1)+
                   (input_to_bottom_hidden * self.output3_w2))
        output4 = ((input_to_top_hidden * self.output4_w1)+
                   (input_to_bottom_hidden * self.output4_w2))

        output_presoftmax = torch.stack([output1, output2, output3, output4])

        return output_presoftmax

    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        print(f'label_i:{label_i}')
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i[0])

        return loss


class WordEmbeddingWithLinear(L.LightningModule):

    def __init__(self):

        super().__init__()

        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):

        hidden = self.input_to_hidden(input[0])
        output_values = self.hidden_to_output(hidden)

        return output_values

    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        print(f'label_i:{label_i}')
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i[0])

        return loss


if __name__ == "__main__":
    print('start')
    inputs = torch.tensor([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
    labels = torch.tensor([[0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.],
                           [0., 1., 0., 0.]])
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)

    # modelFromScratch = WordEmbeddingFromScrath()

    # print("Befor optimization, the patameters are...")
    # for name, param in modelFromScratch.named_parameters():
    #     print(name, param.data)

    # #DataFrame
    # data = {
    #     "w1": [modelFromScratch.input1_w1.item(),
    #            modelFromScratch.input2_w1.item(),
    #            modelFromScratch.input3_w1.item(),
    #            modelFromScratch.input4_w1.item()],
    #     "w2": [modelFromScratch.input1_w2.item(),
    #            modelFromScratch.input2_w2.item(),
    #            modelFromScratch.input3_w2.item(),
    #            modelFromScratch.input4_w2.item()],
    #     "token": ["Troll2", "is", "great", "Gymkata"],
    #     "input": ["input1", "input2", "input3", "input4"]
    # }
    # df = pd.DataFrame(data)
    # print(df)

    # # #sns
    # # sns.scatterplot(data=df, x="w1", y="w2")
    # # for _ in range(0,4):
    # #     plt.text(df.w1[_], df.w2[_], df.token[_],
    # #             horizontalalignment='left',
    # #             size='medium',
    # #             color='black',
    # #             weight='semibold')
    # # plt.show()

    # #========
    # #training
    # #========
    # trainer = L.Trainer(max_epochs=100)
    # trainer.fit(modelFromScratch, train_dataloaders=dataloader)

    # #after_trained
    # data = {
    #     "w1": [modelFromScratch.input1_w1.item(),
    #            modelFromScratch.input2_w1.item(),
    #            modelFromScratch.input3_w1.item(),
    #            modelFromScratch.input4_w1.item()],
    #     "w2": [modelFromScratch.input1_w2.item(),
    #            modelFromScratch.input2_w2.item(),
    #            modelFromScratch.input3_w2.item(),
    #            modelFromScratch.input4_w2.item()],
    #     "token": ["Troll2", "is", "great", "Gymkata"],
    #     "input": ["input1", "input2", "input3", "input4"]
    # }
    # df = pd.DataFrame(data)
    # print(df)

    # # sns.scatterplot(data=df, x="w1", y="w2")
    # # for _ in range(0,4):
    # #     plt.text(df.w1[_], df.w2[_], df.token[_],
    # #             horizontalalignment='left',
    # #             size='medium',
    # #             color='black',
    # #             weight='semibold')
    # # plt.show()

    # #====
    # #test
    # #====
    # softmax = nn.Softmax(dim=0)

    # print(torch.round(softmax(modelFromScratch(torch.tensor([[1., 0., 0., 0.]]))),
    #                   decimals=2))

    #withLinear
    modelLinear = WordEmbeddingWithLinear()

    data = {
        "w1": modelLinear.input_to_hidden.weight.detach()[0].numpy(),
        "w2": modelLinear.input_to_hidden.weight.detach()[1].numpy(),
        "token": ["Troll2", "is", "great", "Gymkata"],
        "input": ["input1", "input2", "input3", "input4"]
    }
    df = pd.DataFrame(data)
    print(df)

    trainer = L.Trainer(max_epochs=100)
    trainer.fit(modelLinear, train_dataloaders=dataloader)

    data = {
        "w1": modelLinear.input_to_hidden.weight.detach()[0].numpy(),
        "w2": modelLinear.input_to_hidden.weight.detach()[1].numpy(),
        "token": ["Troll2", "is", "great", "Gymkata"],
        "input": ["input1", "input2", "input3", "input4"]
    }
    df = pd.DataFrame(data)
    print(df)

    print(modelLinear.input_to_hidden.weight)

    word_embeddings = nn.Embedding.from_pretrained(modelLinear.input_to_hidden.weight.T)

    print(word_embeddings.weight)
    print(word_embeddings(torch.tensor(0)))

    vocab = {'Troll2':0,
             'is':1,
             'great':2,
             'Gymkata':3}
    print(word_embeddings(torch.tensor(vocab['Troll2'])))
