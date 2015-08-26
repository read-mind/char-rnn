require 'nn'
require 'nngraph'
local model_utils = require 'util.model_utils'
require 'util.OneHot'
require 'util.misc'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

local Recurrent, parent = torch.class('nn.Recurrent', 'nn.Module')

function Recurrent:__init(rnn_type, batch_size, seq_length, input_size, rnn_size, output_size, num_layers, dropout)
  parent.__init(self)
  print('rnn type=' .. rnn_type  .. ' batch_size=' .. batch_size .. ' seq_length=' .. seq_length
  .. ' input_size=' .. input_size .. ' rnn_size=' .. rnn_size  .. ' output_size=' .. output_size
  .. ' num_layers=' .. num_layers .. ' dropout=' .. dropout)
  self.clones = {}
  self.protos = {}
  self.params = torch.Tensor()
  self.grad_params = torch.Tensor()
  self.rnn_state = {}
  self.predictions = {}
  self.batch_size = batch_size
  self.seq_length = seq_length
  self.input_size = input_size
  self.rnn_size = rnn_size
  self.output_size = output_size
  self.num_layers = num_layers
  self.dropout = dropout
  self.drnn_state = {}

  if rnn_type == 'lstm' then
      self.protos.rnn = LSTM.lstm(self.input_size, self.rnn_size, self.output_size, self.num_layers, self.dropout)
  elseif rnn_type == 'gru' then
      self.protos.rnn = GRU.gru(self.input_size, self.rnn_size, self.output_size, self.num_layers, self.dropout)
  elseif rnn_type == 'rnn' then
      self.protos.rnn = RNN.rnn(self.input_size, self.rnn_size, self.output_size, self.num_layers, self.dropout)
  end

  self.params, self.grad_params = model_utils.combine_all_parameters(self.protos.rnn)

  for name,proto in pairs(self.protos) do
      print('cloning ' .. name)
      self.clones[name] = model_utils.clone_many_times(proto, seq_length)
  end

  -- the initial state of the cell/hidden states
  self.init_state = {}
  for L=1,num_layers do
      local h_init = torch.zeros(self.batch_size, self.rnn_size)
      table.insert(self.init_state, h_init:clone())
      if rnn_type == 'lstm' then
          table.insert(self.init_state, h_init:clone())
      end
  end
end


function Recurrent:updateOutput(input)
    self.rnn_state = {[0] = self.init_state}
    self.predictions = {}
    local seq_length = input:size(2)
    for t=1,seq_length do
        self.clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = self.clones.rnn[t]:forward{input[{{}, t, {}}], unpack(self.rnn_state[t-1])}
        self.rnn_state[t] = {}
        for i=1,#self.init_state do table.insert(self.rnn_state[t], lst[i]) end -- extract the state, without output
        self.predictions[t] = lst[#lst] -- last element is the prediction
    end
    
    self.output:resize(input:size(1),input:size(2), self.output_size)
    for t=1,seq_length do
      self.output[{{},t,{}}] = self.predictions[t]
    end
  return self.output
end


function Recurrent:updateGradInput(input, gradOutput)
  self.gradInput:resize(input:size())
  local seq_length = input:size(2)
  -- initialize gradient at time t to be zeros (there's no influence from future)
  self.drnn_state = {[seq_length] = clone_list(self.init_state, true)} -- true also zeros the clones
  for t=seq_length,1,-1 do
      local doutput_t = gradOutput[{{}, t, {}}]
      table.insert(self.drnn_state[t], doutput_t)
      local dlst = self.clones.rnn[t]:backward({input[{{}, t, {}}], unpack(self.rnn_state[t-1])}, self.drnn_state[t])
      self.drnn_state[t-1] = {}
      for k,v in pairs(dlst) do
          self.drnn_state[t-1][k-1] = v
      end
  end

  for t=seq_length,1,-1 do
    self.gradInput[{{},t, {}}] = self.drnn_state[t-1][0]
  end
  return self.gradInput
end

function Recurrent:accGradParameters(input, gradOutput, scale)
  local seq_length = input:size(2)
  for t=seq_length,1,-1 do
      self.clones.rnn[t]:accGradParameters({input[{{}, t, {}}], unpack(self.rnn_state[t-1])}, self.drnn_state[t])
  end
end

function Recurrent:parameters()
  return self.params, self.grad_params
end

local model = nn.Sequential()
model:add(nn.Recurrent('lstm', 2, 10, 5, 256, 5, 2, 0))
model:add(nn.Sigmoid())
--model = nn.Recurrent('lstm', 2, 10, 5, 256, 5, 2, 0)

p, gp = model:getParameters()
--p, gp = model:parameters()
print('number of parameters in the model: ' .. p:nElement())

p:uniform(-0.08, 0.08) -- small numbers uniform
print('finish Recurrent buiding')
criterion = nn.MSECriterion()

sys.sleep(2)

my_input = torch.rand(2,10,5):bernoulli()
my_input = my_input:double()
print('input')
print(my_input)


my_label = torch.zeros(2,10,5)
my_label[{{},{2,10},{}}] = my_input[{{},{1,9},{}}]
print('my_label')
print(my_label)
sys.sleep(1)


my_output = model:forward(my_input)
print('output')
print(my_output)


print('gp1')
print(gp[{{1,10}}])

local lr = 0.1
for i=1,1000000 do
  my_output = model:forward(my_input)
  err = criterion:forward(my_output, my_label)
  print(err/my_input:size(2))
  d_out = criterion:backward(my_output, my_label)
  model:backward(my_input,d_out)

  -- clip gradient element-wise
  gp:clamp(-5, 5)
  p:add(-1*lr, gp)

  --sys.sleep(1)
  if i % 1000 == 0 then 
    lr = lr * 0.5
  end
  if i % 10 == 0 then
    --print(my_output)
    print('p2')
    print(p[{{1,10}}])
    print('gp2')
    print(gp[{{1,10}}])
  end
end

