import torch.nn as nn

# Neural Network definition
class SimpleNN(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, depth=2, dropout_prob=0.0, activation_fn=nn.ReLU):
        super(SimpleNN, self).__init__()
        self.fc_in = nn.Linear(in_features, hidden_size)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(activation_fn())
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.Dropout(p=dropout_prob))
            self.hidden_layers.append(activation_fn())
        
        self.fc_out = nn.Linear(hidden_size, out_features)
        
    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc_out(x)
        return x