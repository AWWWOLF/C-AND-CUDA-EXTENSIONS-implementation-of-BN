import torch

#导入我的模型
import BatchNorm

class BNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, running_mean, running_var, eps, momentum, mode):
        outputs = BatchNorm.forward(X, gamma, beta, running_mean, running_var, eps, momentum, mode)
        output = outputs[0]
        var = outputs[1]
        out_ = outputs[2]
        gamma = outputs[3]
        varibles = [var, out_, gamma]
        ctx.save_for_backward(*varibles)
        return output

    @staticmethod
    def backward(ctx, dout):
        outputs = BatchNorm.backward(dout.contiguous(), *ctx.saved_variables)
        dX, dgamma, dbeta = outputs
        return dX, dgamma, dbeta, None, None, None, None, None

class BN(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BN, self).__init__()
        self.num_features = num_features
        self.eps = torch.tensor(eps)
        self.momentum = torch.tensor(momentum)
        self.gamma = torch.empty(num_features)
        self.beta = torch.empty(num_features)
        self.running_mean = torch.empty(num_features)
        self.running_var = torch.empty(num_features)
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.mode = torch.tensor([1, 1, 1])
        else:
            self.mode = torch.tensor([1])

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.beta.zero_()
            self.gamma.fill_(1)

    def forward(self, X):
        if (len(X.shape) == 2):
            X = torch.unsqueeze(X, 2)
            X = torch.unsqueeze(X, 3)
        self.gamma = self.gamma.to(X.device)
        self.beta = self.beta.to(X.device)
        self.running_mean = self.running_mean.to(X.device)
        self.running_var = self.running_var.to(X.device)
        self.eps = self.eps.to(X.device)
        self.momentum = self.momentum.to(X.device)
        self.mode = self.mode.to(X.device)
        return BNFunction.apply(X, self.gamma, self.beta, self.running_mean, self.running_var, self.eps, self.momentum, self.mode)