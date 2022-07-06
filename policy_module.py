import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np
import collections
import gym

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cont_Deterministic_Policy_Module(nn.Module): #for continuous
  def __init__(self, i_size, hidden_sizes, o_size):
    super(Cont_Deterministic_Policy_Module, self).__init__()
    #create N layers of mlp for output
    layers=[]
    len_h=len(hidden_sizes)
    relu=nn.ReLU()

    first=nn.Linear(i_size, hidden_sizes[0]).to(device)
    f=i_size
    nn.init.uniform_(first.weight, -1/np.sqrt(f), 1/np.sqrt(f))
    nn.init.zeros_(first.bias)
    layers.append(first)
    layers.append(relu)
    for h_idx in range(len_h-1):
      layer=nn.Linear(hidden_sizes[h_idx], hidden_sizes[h_idx+1]).to(device)
      f=hidden_sizes[h_idx]
      nn.init.uniform_(layer.weight, -1/np.sqrt(f), 1/np.sqrt(f))
      nn.init.zeros_(layer.bias)
      layers.append(layer)
      layers.append(relu)
    last=nn.Linear(hidden_sizes[-1], o_size).to(device)
    nn.init.uniform_(last.weight, -(3e-3), 3e-3)
    nn.init.zeros_(last.bias)
    layers.append(last)
    layers.append(nn.Tanh())
    #last activ ftn: tanh
    self.linear_layers=nn.Sequential(*list(layers))
    self.w_sizes, self.b_sizes=self.get_parameter_sizes()
      
  def show_params(self):
    for idx, params in enumerate(self.parameters()):
      print("Idx: {:d}".format(idx))
      print(params.size())
      print(params)
    return
  
  def get_parameter_sizes(self): #initialize linear layers in this part
    w_sizes=[]
    b_sizes=[]
    for idx, element in enumerate(self.linear_layers):
      if isinstance(element, nn.Linear):
        w_s=element.weight.size()
        b_s=element.bias.size()
        w_sizes.append(w_s)
        b_sizes.append(b_s)
    return w_sizes, b_sizes
  
  def vectorize_parameters(self):
    parameter_vector=torch.Tensor([]).to(device)
    for param in self.parameters():
      p=param.reshape(-1,1)
      parameter_vector=torch.cat((parameter_vector, p), dim=0)
    return parameter_vector.squeeze(dim=1)
  
  def vectorize_gradient(self):
    gradient_vector=torch.Tensor([]).to(device)
    for param in self.parameters():
      grad=param.grad
      g_v=grad.reshape(-1,1)
      gradient_vector=torch.cat((gradient_vector, g_v), dim=0)
    return gradient_vector.squeeze(dim=1)

  def inherit_parameters(self, parameter_vector):
    #size must be identical, input in vectorized form
    vector_idx=0
    #extract weight, bias data
    weights=[]
    biases=[]
    for sz_idx, w_size in enumerate(self.w_sizes):
      w_length=w_size[0]*w_size[1]
      weight=parameter_vector[vector_idx:vector_idx+w_length]
      weight=weight.reshape(w_size[0], w_size[1])
      weights.append(weight)
      vector_idx=vector_idx+w_length
      b_length=self.b_sizes[sz_idx][0]
      bias=parameter_vector[vector_idx:vector_idx+b_length]
      bias=bias.reshape(-1)
      biases.append(bias)
      vector_idx=vector_idx+b_length
    #overwrite parameters
    linear_idx=0
    for element in self.linear_layers:
      if isinstance(element, nn.Linear):
        element.weight=nn.Parameter(weights[linear_idx])
        element.bias=nn.Parameter(biases[linear_idx])
        linear_idx+=1
    return
  
  #parameter forward-> forward pass using arbitrary parameters w/o actually changing parameters
  def parameter_forward(self, observation, parameter_vector):
    relu=nn.ReLU()
    vector_idx=0
    #extract weight, bias data
    z=observation
    for sz_idx, w_size in enumerate(self.w_sizes):
      w_length=w_size[0]*w_size[1]
      weight=parameter_vector[vector_idx:vector_idx+w_length]
      weight=weight.reshape(w_size[0], w_size[1])
      vector_idx=vector_idx+w_length
      b_length=self.b_sizes[sz_idx][0]
      bias=parameter_vector[vector_idx:vector_idx+b_length]
      bias=bias.reshape(-1)
      vector_idx=vector_idx+b_length
      z=weight.matmul(z)+bias
      if sz_idx<len(self.w_sizes)-1:
        z=relu(z)
    return z #return logits

  def forward(self, observation):
    action=self.linear_layers(observation)
    return action