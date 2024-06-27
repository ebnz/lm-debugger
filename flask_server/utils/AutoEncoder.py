import torch

class AutoEncoderTM(torch.nn.Module):
    def __init__(self, n, m):
        super().__init__()

        self.n = n
        self.m = m
        
        self.bias_encoder = torch.nn.Parameter(torch.rand(m))
        self.bias_decoder = torch.nn.Parameter(torch.rand(n))

        self.weight_encoder = torch.nn.Parameter(torch.rand(n, m))
        self.weight_decoder = torch.nn.Parameter(torch.rand(m, n))

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_bar = x - self.bias_decoder
        f = self.relu(x_bar @ self.weight_encoder + self.bias_encoder)
        x_hat = f @ self.weight_decoder + self.bias_decoder

        return x_hat, f

def AELossTM_reconstruction(X, X_hat):
    return torch.mean(torch.norm(X - X_hat, dim=1, p=2)**2)

def AELossTM_sparsity(f, lam=1):
    return torch.mean(lam * torch.norm(f, dim=1, p=1))

def AELossTM(X, X_hat, f, lam=1):
    return AELossTM_reconstruction(X, X_hat) + AELossTM_sparsity(f, lam=lam)
    #return torch.mean(torch.norm(X - X_hat, dim=1, p=2)**2 + lam * torch.norm(f, dim=1, p=1))


class AutoEncoderNN(torch.nn.Module):
    def __init__(self, n, m):
        super().__init__()

        self.n = n
        self.m = m
        
        self.bias_encoder = torch.nn.Parameter(torch.zeros(m))
        self.bias_decoder = torch.nn.Parameter(torch.zeros(n))

        self.weight_encoder = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.zeros(self.n, self.m)))
        self.weight_decoder = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.zeros(self.m, self.n)))

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_bar = x - self.bias_decoder
        f = self.relu(x_bar @ self.weight_encoder + self.bias_encoder)
        x_hat = f @ self.weight_decoder + self.bias_decoder

        return x_hat, f

def AELossNN_reconstruction(X, X_hat):
    return torch.mean(torch.sum((X - X_hat)**2, dim=1))

def AELossNN_sparsity(f, lam=1):
    return torch.sum(lam * torch.abs(f))

def AELossNN(X, X_hat, f, lam=1):
    return AELossNN_reconstruction(X, X_hat) + AELossNN_sparsity(f, lam=lam)


class AutoEncoderSC(torch.nn.Module):
    def __init__(self, n, m, reinit_dead_neurons=False):
        super().__init__()

        self.n = n
        self.m = m
        self.reinit_dead_neurons = reinit_dead_neurons
        
        self.training_steps = 0
        self.summed_f_activations = torch.zeros((m))

        self.weight = torch.nn.Parameter(torch.rand(m, n))
        self.bias = torch.nn.Parameter(torch.rand(m))

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        f = self.relu(x @ self.weight.T + self.bias)
        x_hat = f @ self.weight

        if self.reinit_dead_neurons:
            self.training_steps += 1
            self.summed_f_activations += torch.sum(f.detach().cpu(), dim=0)

            if self.training_steps == 12_500:
                for feature_index in range(len(self.summed_f_activations)):
                    if self.summed_f_activations[feature_index] == 0.0:
                        self.summed_f_activations[feature_index] = torch.zeros((1))
                        self.weight[feature_index] = torch.rand(self.n).to(self.weight.device)
                        self.bias[feature_index] = torch.rand(1).to(self.bias.device)

                self.training_steps = 0
                self.summed_f_activations = torch.zeros((self.m))
        
        return x_hat, f

def AELossSC(X, X_hat, f, lam=1):
    return torch.mean(torch.norm((X - X_hat), p=2, dim=1)**2 + lam * torch.norm(f, p=1, dim=1))