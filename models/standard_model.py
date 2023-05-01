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
       # x = self.fc_final(x)
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

if __name__ == '__main__':
    torch.random.manual_seed(0)
    batch_size = 100
    dim_in = 2
    model = Standard_FNN(dim_in, 2, [2, 4])
    x = torch.normal(0, 0.2, (batch_size, dim_in))
    x[:batch_size // 2, 0] = x[:batch_size // 2, 0] + 2
    y = torch.ones((batch_size,))
    y[:batch_size // 2] = 0
    print(f"Centered at (2, 0) with label {y[0].long().item()}", x[:5])
    print(f"Centered at (0, 0) with label {y[-1].long().item()}", x[-5:])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    print(model)
    for epoch in range(500):
        optimizer.zero_grad()
        loss = model.get_loss(x, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(loss)

    model.eval()
    with torch.no_grad():
        y_pred = model.forward(x).argmax(dim=-1)
        print("Acc:", torch.mean((y_pred == y).float()).item())
