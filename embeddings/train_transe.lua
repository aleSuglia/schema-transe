require "dataset.lua";
require "samplers.lua";
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
        - train_filename: filename of triples data
        - triple_delimiter: delimiter of triples elements
        - model_filename: filename of learnt model
        - kb_index_filename: filename of knowledge base index
        - optim_method: optimization method identifier used in optim package
        - training_params: parameters of training method
        - batch_size: number of training examples in a batch
        - num_epochs: number of training epochs
        - embeddings_size: size of entity and relation embeddings
        - margin: margin for the hinge criterion
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

print("-- Loading knowledge base index: " .. conf_data["kb_index_filename"])
local kb_index = torch.load(conf_data["kb_index_filename"])

print("-- Loading triples data: " .. conf_data["train_filename"])
local triples = read_triples(conf_data["train_filename"], conf_data["triple_delimiter"], kb_index)

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

local embeddings_size = conf_data["embeddings_size"]
local margin = conf_data["margin"]
local batch_size = conf_data["batch_size"]
local num_epochs = conf_data["num_epochs"]
local conf_sampler = conf_data["sampler"]
local sampler = create_sampler(conf_sampler["id"])
local num_corrupted = conf_sampler["params"]["size"]

local training_params = {}
for k, v in pairs(conf_data["training_params"]) do
   training_params[k] = v
end

local num_triples = triples:size(1)
local num_entities = #kb_index["entity2id"]
local num_relations = #kb_index["relation2id"]

print("Dataset stats:")
print("Number of triples:\t" .. num_triples)
print("Number of entities:\t" .. num_entities)
print("Number of relations:\t" .. num_relations)

print("-- Building TransE model")
local model = build_transe_model(num_entities, num_relations, embeddings_size)

local criterion = nn.MarginRankingCriterion(margin)
criterion.sizeAverage = false

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
        local correct_triples_batch = torch.Tensor(batch_size * num_corrupted, 3)
        local corrupted_triples_batch = torch.Tensor(batch_size * num_corrupted, 3)
        local targets = torch.Tensor(batch_size * num_corrupted, 1):fill(-1)

        if conf_data["cuda"] then
            correct_triples_batch = correct_triples_batch:cuda()
            corrupted_triples_batch = corrupted_triples_batch:cuda()
            targets = targets:cuda()
        end

      corrupted_triples_batch = sampler(triples:index(1, v), kb_index, conf_sampler["params"])

      for i=1, v:size(1) do
         local triple_id = v[i]
         local current_triple = triples[triple_id]
         correct_triples_batch[{{(i-1)*num_corrupted+1, i*num_corrupted}, {}}] =
            torch.repeatTensor(current_triple, num_corrupted, 1)
      end

      local inputs = {
            {
                correct_triples_batch[{{}, 1}],
                correct_triples_batch[{{}, 2}],
                correct_triples_batch[{{}, 3}]},
            {
                corrupted_triples_batch[{{}, 1}],
                corrupted_triples_batch[{{}, 2}],
                corrupted_triples_batch[{{}, 3}]
            }
        }

      -- callback that does a single batch optimization step
      local batch_optimize = function(x)
         -- get new parameters
         if x ~= params then
            params:copy(x)
         end

         -- reset gradients
         grad_params:zero()
            
         -- normalize entities lookup weights
         local entities_lookup = model:get(1):get(1):get(1):get(1)
         normalize_lookup_table(entities_lookup, 2)

         -- backward propagation
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         if conf_data["training_params"]["weightDecay"] then
            local coefL2 = conf_data["training_params"]["weightDecay"]
            local norm, sign = torch.norm, torch.sign
            f = f + coefL2 * norm(params, 2)^2/2
            grad_params:add(params:clone():mul(coefL2))
         end

         -- return f and df/dX
         return f, grad_params
      end

      -- optimize on current mini-batch
      local _, fs = optim_method(batch_optimize, params, training_params)

      -- evaluate current loss function value
      local current_cost = fs[1]
      average_cost = average_cost + current_cost

      -- show custom progress bar
      progress(t, #indices, current_cost)
   end

   -- evaluate average cost per epoch
   local average_cost = round(average_cost / #indices, 4)
   print("Average cost per epoch: " .. average_cost)
end

print("-- Saving final model")
torch.save(conf_data["model_filename"], model)
