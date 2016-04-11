require "dataset.lua";
require "models.lua";
require "progress.lua";
require "xlua";
require "optim";
file = require "pl.file";
cjson = require "cjson";

math.randomseed(12345)
torch.setdefaulttensortype("torch.FloatTensor")
torch.manualSeed(12345)

function round(val, decimal)
    local exp = decimal and 10 ^ decimal or 1
    return math.ceil(val * exp - 0.5) / exp
end

local title = "### TransE ###"

print(title)

--[[
    JSON configuration file parameters:
        - triples_filename: filename of triples in CSV format
        - model_filename: filename of the learnt model
        - optim_method: optimization method identifier used in optim package
        - training_params: parameters of the training method
        - batch_size: number of training examples in a batch
        - num_epochs: number of training epochs
        - entities_embeddings_size: size of entity embeddings
        - relations_embeddings_size: size of relation embeddings
        - margin: margin for the Hinge criterion
        - cuda: true enables CUDA, otherwise false
--]]

local cmd = torch.CmdLine()
cmd:text()
cmd:text(title)
cmd:text()
cmd:text("Options:")
cmd:option("-config", "", "Filename of JSON training parameters")
cmd:text()

local params = cmd:parse(arg)

local conf_data = cjson.decode(file.read(params.config))

print("-- Loading triples data: " .. conf_data["triples_filename"])
local triples_data = read_triples_data(conf_data["triples_filename"])

local optim_method_id = conf_data["optim_method"]
local optim_method

if optim_method_id == "sgd" then
    optim_method = optim.sgd
elseif optim_method_id == "adadelta" then
    optim_method = optim.adadelta
elseif optim_method_id == "adagrad" then
    optim_method = optim.adagrad
elseif optim_method_id == "adam" then
    optim_method = optim.adam
elseif optim_method_id == "rmsprop" then
    optim_method = optim.rmsprop
else
    print("Invalid training method: " .. optim_method_id)
end

local entities_embeddings_size = conf_data["entities_embeddings_size"]
local relations_embeddings_size = conf_data["relations_embeddings_size"]
local margin = conf_data["margin"]
local batch_size = conf_data["batch_size"]
local num_epochs = conf_data["num_epochs"]

local training_params = {}

for k, v in pairs(conf_data["training_params"]) do
   training_params[k] = v
end

local num_triples = triples_data["triples"]:size(1)

print("Dataset stats:")
print("Number of triples:\t" .. num_triples)
print("Number of entities:\t" .. #triples_data["entity2id"])
print("Number of relations:\t" .. #triples_data["relation2id"])

print("-- Building TransE model")
local model = build_transe_model(triples_data, entities_embeddings_size, relations_embeddings_size)

local criterion = nn.HingeEmbeddingCriterion(margin)

if conf_data["cuda"] then
   require "cutorch";
   require "cunn";

   cutorch.setDevice(1)
   cutorch.manualSeed(12345)
   model = model:cuda()
   criterion = criterion:cuda()   
end

-- get model parameters
local params, grad_params = model:getParameters()

print("-- Training model using " .. optim_method_id)

for e = 1, num_epochs do
   -- shuffle and split training examples in batches
   local indices = torch.randperm(num_triples):long():split(batch_size)

   -- remove last element so that all the batches have equal size
   indices[#indices] = nil

   print("==> doing epoch on training data:")
   print("==> online epoch # " .. e .. " [batchSize = " .. batch_size .. "]")

   local average_cost = 0

   for t, v in ipairs(indices) do
      local correct_triples_batch = torch.Tensor(2 * batch_size, 3):zero()
      local corrupted_triples_batch = torch.Tensor(2 * batch_size, 3):zero()
      local targets = torch.Tensor(2 * batch_size):zero()

      if conf_data["cuda"] then
         correct_triples_batch = correct_triples_batch:cuda()
         corrupted_triples_batch = corrupted_triples_batch:cuda()
         targets = targets:cuda()
      end
      
      local j = 1

      for i=1, v:size(1) do
         local triple_id = v[i]
         local current_triple = triples_data["triples"][triple_id]
         local corrupted_triple = sample_corrupted_triple(triples_data, current_triple)

         correct_triples_batch[{j, {}}] = current_triple
         corrupted_triples_batch[{j, {}}] = current_triple
         targets[j] = 1
         j = j + 1

         correct_triples_batch[{j, {}}] = current_triple
         corrupted_triples_batch[{j, {}}] = corrupted_triple
         targets[j] = -1
         j = j + 1
      end

      local inputs = {correct_triples_batch, corrupted_triples_batch}
    
      -- callback that does a single batch optimization step
      local batch_optimize = function(x)
         -- get new parameters
         if x ~= params then
            params:copy(x)
         end

         -- reset gradients
         grad_params:zero()

         -- backward propagation
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)
         
         -- return f and df/dX
         return f, grad_params
      end

      -- optimize on current mini-batch
      --local _, fs = optim_method(batch_optimize, params, training_params)

      current_cost = criterion:forward(model:forward(inputs), targets)
      model:zeroGradParameters()
      model:backward(inputs, criterion:backward(model.output, targets))
      model:updateParameters(0.01)
      -- evaluate current loss function value
      --local current_cost = fs[1]
      average_cost = average_cost + current_cost
      
      -- show custom progress bar
      progress(t, #indices, current_cost)
   end

   -- evaluate average cost per epoch
   average_cost = round(average_cost / #indices, 4)
   print("Average cost per epoch: " .. average_cost)
  
end

print("-- Saving final model")
torch.save(conf_data["model_filename"], model)
