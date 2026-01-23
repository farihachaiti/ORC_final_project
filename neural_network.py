import torch
import torch.nn as nn

device='cuda' if torch.cuda.is_available() else 'cpu'
class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, input_size, hidden_size, output_size, activation=nn.Tanh(), ub=None):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )
        self.ub = ub if ub is not None else torch.tensor(1, dtype=torch.float32) # upper bound of the output layer
        self.initialize_weights()

    def forward(self, x):
        
        x = x.mT
        x = x.squeeze(-1)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(next(self.parameters()).device)  # Move input to model's device

        out = self.linear_stack(x) * self.ub
        return out
   
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias) 

    def create_casadi_function(self, model_name, NN_DIR, input_size, load_weights):
        from casadi import MX, Function, vertcat
        import l4casadi as l4c

        # if load_weights is True, we load the neural-network weights from a ".pt" file
        if load_weights:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            nn_name = f'{NN_DIR}/nn_{model_name}/{model_name}_model.pt'

            try:
                # Try loading as a TorchScript model
                self.ts_model = torch.jit.load(nn_name, map_location=device)
                self.ts_model.eval()
                self.is_torchscript = True

            except Exception:
                # Fallback: try loading as state_dict
                checkpoint = torch.load(nn_name, map_location=device)

                if 'model' in checkpoint:
                    self.load_state_dict(checkpoint['model'])
                else:
                    self.load_state_dict(checkpoint)

                self.to(device)
                self.eval()
                self.is_torchscript = False

        
        # Create input symbols with the correct shape
        state = MX.sym("x", input_size)
        # Create the L4CasADi model
        self.l4c_model = l4c.L4CasADi(
            self,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        
        # Ensure the input has the right shape for the model
       
            
        # Create the model function
        self.nn_model = self.l4c_model(state)
        
        # This is the function that you can use in a casadi problem
        nn_func = Function('nn_func', [state], [self.nn_model])
        
        return nn_func