import deepspeed
import argparse

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--local_rankk', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--ds', type=str, 
                    help='local rank passed from distributed launcher')
# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()


import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        
        # Define a simple feed-forward neural network with one hidden layer
        self.fc1 = nn.Linear(in_features=10, out_features=50)  # Input layer with 10 features
        self.fc2 = nn.Linear(in_features=50, out_features=1)   # Output layer with 1 feature
        
    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model
model = DummyModel()

model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     config=cmd_args.ds)
print(model_engine)