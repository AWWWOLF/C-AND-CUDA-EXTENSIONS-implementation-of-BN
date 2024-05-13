#include <torch/torch.h>
#include <iostream>
#include <vector>

std::vector<torch::Tensor> forward(
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor eps,
    torch::Tensor momentum,
    torch::Tensor mode
    ){
        // 获取输入张量的尺寸
        const auto M = X.size(0);
        const auto D = X.size(1);
        const auto H = X.size(2);
        const auto W = X.size(3);
        //初始化输出和缓存
        auto output = torch::empty({M, D, H, W}, X.options());
        auto out_ = torch::empty({M, D, H, W}, X.options());
        auto sample_mean = torch::empty(D, X.options());
        auto sample_var = torch::empty(D, X.options());
        auto var = torch::empty(D, X.options());
        //扩充维度
        auto running_mean_ = running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3);
        auto running_var_ = running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3);
        auto gamma_ = gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3);
        auto beta_ = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3);
        if (mode.numel() == 3)
        {   // 计算样本均值和样本方差
            sample_mean = torch::mean(X, {0, 2, 3});
            sample_var = torch::var(X, {0, 2, 3}, false);
            var = 1 / torch::sqrt(sample_var + eps);

            //扩充维度
            auto sample_mean_ = sample_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3);
            auto sample_var_ = sample_var.unsqueeze(0).unsqueeze(2).unsqueeze(3);

            // 计算归一化张量
            out_= (X - sample_mean_) / (torch::sqrt(sample_var_ + eps));
            output = gamma_ * out_ + beta_;

            // 更新运行均值和运行方差
            auto n = M * H * W;
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean;
            running_var = momentum * running_var * n / (n-1) + (1 - momentum) * sample_var;
        }
        else
        {   //不处于训练模式就固定running_mean和running_var的值
            //计算归一化张量
            auto X_hat =(X - running_mean_) / (torch::sqrt(running_var_ + eps));
            output = gamma_ * X_hat + beta_;
        }
        return {output, var, out_, gamma};
    }

std::vector<torch::Tensor> backward(
        torch::Tensor dout,
        torch::Tensor var,
        torch::Tensor out_,
        torch::Tensor gamma)
    {   //计算梯度大小
        auto dbeta = dout.sum((0, 2, 3), true);
        auto dgamma = (out_ * dout).sum((0, 2, 3), true);

        //扩充张量
        auto gamma_ = gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3);
        auto var_ = var.unsqueeze(0).unsqueeze(2).unsqueeze(3);

        //计算梯度
        auto dY = dout * gamma_;
        auto dX = var_ * (dY - torch::mean(dY, {0, 2, 3}) - out_ * torch::mean(dY * out_, {0, 2, 3}));
        return {dX, dgamma, dbeta};
    }

// 导出函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "BN forward");
  m.def("backward", &backward, "BN backward");
}