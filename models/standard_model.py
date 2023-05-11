import torch
import torch.nn.functional as F


class Standard_FNN(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens: list, activation=F.relu):
        super(Standard_FNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.activation = activation

        for i, num_hidden in enumerate(num_hiddens):
            setattr(self, 'fc{}.linear'.format(i),
                    torch.nn.Linear(num_inputs, num_hidden))
            num_inputs = num_hidden

        setattr(self, 'fc_final.{}'.format("linear"),
                torch.nn.Linear(num_inputs, num_outputs))
        self.loss_func = torch.nn.BCELoss(reduce=False)

    def forward_except_last(self, x):
        for i, num_hidden in enumerate(self.num_hiddens):
            x = getattr(self, 'fc{}.linear'.format(i))(x)
            x = self.activation(x)
        return x

    def forward(self, x):
        x = x.float()
        x = self.forward_except_last(x)
        x = getattr(self, 'fc_final.linear')(x)
        return x

    def get_loss(self, x, y):
        '''
            x is data
            y is ground-truth label
        '''
        x = x.float()
        output = self.forward(x)
        output = torch.sigmoid(output[:, 1] - output[:, 0])
        loss = self.loss_func(output, y.float())
        return loss.mean()
