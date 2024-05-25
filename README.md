# C++ AND CUDA Extensions Implementation of BN With Pytorch
### This is a homework from my Deeplearning lesson. It may not be a professional one.  
### These code realize the function of Batch Normalization with c++ or c++&CUDA.  
### When you firstly download it in to your computer, please enter the file, run:  
```bash env.sh```  
### then, run:  
```conda activate BNCUDA```   
### then if you want to use c++ extension, run:  
```python setup.py install```  
### or you want to use c++/CUDA extenxion, run:  
```python setup_cuda.py install```  
### After successfully installing, you can run:  
```python test.py```  
### to see the different performances between c++, c++/CUDA and Pytorch.  
### References:  
<https://pytorch.org/tutorials/advanced/cpp_extension.html>  
<https://blog.csdn.net/qq_27370437/article/details/119569007?spm=1001.2014.3001.5506>  
<https://www.cnblogs.com/yanghailin/p/12901586.html>  
<https://zhuanlan.zhihu.com/p/659430546>  
<https://zhuanlan.zhihu.com/p/544864997>  
<https://arxiv.org/abs/1502.03167>  
<https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>  
### PS: My CUDA version is 12.0, but 12.1 can still run correctly, see:  
<https://discuss.pytorch.org/t/trying-and-failing-to-install-pytorch-for-cuda-12-0/186194/3>
# BatchNorm-Extension
