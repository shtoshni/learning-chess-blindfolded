import torch
import torch.nn as nn


class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_hidden_layers=1, bias=False, drop_module=None):
        super(MyMLP, self).__init__()
        self.layer_list = []

        self.activation = nn.ReLU()
        self.drop_module = drop_module
        self.num_hidden_layers = num_hidden_layers

        cur_output_size = input_size
        for i in range(num_hidden_layers):
            self.layer_list.append(nn.Linear(cur_output_size, hidden_size, bias=bias))
            self.layer_list.append(self.activation)
            if self.drop_module is not None:
                self.layer_list.append(self.drop_module)
            cur_output_size = hidden_size

        self.layer_list.append(nn.Linear(cur_output_size, output_size, bias=bias))
        self.fc_layers = nn.Sequential(*self.layer_list)

    def forward(self, mlp_input):
        output = self.fc_layers(mlp_input)
        return output
