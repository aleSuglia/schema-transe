require "dataset.lua";
require "nn";
require "xlua";
tds = require "tds";
file = require "pl.file";
cjson = require "cjson";

function eval_triple_ranking(triple_batch, num_entities, outputs, inverted_relations, topn)
   local triple_batch_subj = triple_batch[{{1, num_entities}, {}}]
   local triple_batch_obj = torch.Tensor(num_entities, 3)
   triple_batch_obj[{{1}, {}}] = triple_batch[{{1}, {}}]
   triple_batch_obj[{{2, num_entities}, {}}] = triple_batch[{{num_entities+1, 2*num_entities-1}, {}}]

   local outputs_subj = outputs[{{1, num_entities}}]
   local outputs_obj = torch.Tensor(num_entities)
   outputs_obj[1] = outputs[1]
   outputs_obj[{{2, num_entities}}] = outputs[{{num_entities+1, 2*num_entities-1}}]
   
   local current_relation = triple_batch[1][2]

   local subj_hits = 0
   local correct_subj_rank
   local y, indexes = torch.sort(outputs_subj)
   for i=1, indexes:size(1) do
       local current_triple = triple_batch_subj[indexes[i]]
       local subj = current_triple[1]
       local rel = current_triple[2]
       local obj = current_triple[3]
        if subj == triple_batch_subj[1][1] and
          rel == triple_batch_subj[1][2] and
          obj == triple_batch_subj[1][3] then
          correct_subj_rank = i
          if i <= topn then
              subj_hits = subj_hits + 1
          end
        end
   end

   local obj_hits = 0
   local correct_obj_rank
   local y, indexes = torch.sort(outputs_obj)
   for i=1, indexes:size(1) do
       local current_triple = triple_batch_obj[indexes[i]]
       local subj = current_triple[1]
       local rel = current_triple[2]
       local obj = current_triple[3]
       if subj == triple_batch_obj[1][1] and
           rel == triple_batch_obj[1][2] and
           obj == triple_batch_obj[1][3] then
           correct_obj_rank = i
           if i <= topn then
               obj_hits = obj_hits + 1
           end
       end
   end

   return subj_hits, obj_hits, correct_subj_rank, correct_obj_rank
end

function build_inverted_relations(triples)
   local inverted_relations = tds.Hash()

   for i=1, triples:size(1) do
        local current_triple = triples[i]
        local subj = current_triple[1]
        local rel = current_triple[2]
        local obj = current_triple[3]
        
        if not inverted_relations[subj] then
            inverted_relations[subj] = tds.Hash()
        end
        
        if not inverted_relations[subj][obj] then
            inverted_relations[subj][obj] = tds.Hash()
        end
        
        if not inverted_relations[subj][obj][rel] then
            inverted_relations[subj][obj][rel] = tds.Hash()
        end
        
        inverted_relations[subj][obj][rel] = true
   end

   return inverted_relations
end

function evaluate_predictions(model, test_set, kb_index, topn, batch_size, cuda)
   local num_entities = #kb_index["entity2id"]
   local inverted_relations = build_inverted_relations(test_set)
   local average_subj_hits = 0
   local average_obj_hits = 0
   local mean_subj_rank = 0
   local mean_obj_rank = 0
   local mean_reciprocal_subj_rank = 0
   local mean_reciprocal_obj_rank = 0

   for i=1, test_set:size(1) do
      xlua.progress(i, test_set:size(1))

      local current_triple_batch = torch.Tensor(2*num_entities-1, 3)
        
      if cuda then
          current_triple_batch = current_triple_batch:cuda()
      end
        
      local current_triple = test_set[i]
      local entities_ids_nosubj = torch.Tensor(num_entities-1)
      local entities_ids_noobj = torch.Tensor(num_entities-1)
      
      -- get entities ids
      local key_index_nosubj = 1      
      local key_index_noobj = 1
      for k, _ in pairs(kb_index["id2entity"]) do
         if k ~= current_triple[1] then
            entities_ids_nosubj[key_index_nosubj] = k
            key_index_nosubj = key_index_nosubj + 1
         end

         if k ~= current_triple[3] then
            entities_ids_noobj[key_index_noobj] = k
            key_index_noobj = key_index_noobj + 1
         end
      end
      
      -- initialize correct triple
      current_triple_batch[{1, {}}] = current_triple

      -- initialize relation and object columns in the range (2, num_entities)
      -- using current triple relation and object ids
      current_triple_batch[{{2, num_entities}, 2}]:fill(current_triple[2])
      current_triple_batch[{{2, num_entities}, 3}]:fill(current_triple[3])
        
      -- initialize the subject column values in the range (2, num_entities)
      -- using corrupted subject id 
      current_triple_batch[{{2, num_entities}, 1}] = entities_ids_nosubj

      -- initialize relation and object columns in the range (num_entities+1, 2*num_entities)
      -- using current triple subject and relation ids
      current_triple_batch[{{num_entities+1, 2*num_entities-1}, 1}]:fill(current_triple[1])
      current_triple_batch[{{num_entities+1, 2*num_entities-1}, 2}]:fill(current_triple[2])
        
      -- initialize the subject column values in the range (2, num_entities)
      -- using corrupted object id 
      current_triple_batch[{{num_entities+1, 2*num_entities-1}, 3}] = entities_ids_noobj  

      local splitted_current_triple_batch = current_triple_batch:split(batch_size)
      local outputs = torch.Tensor(current_triple_batch:size(1))

      local k = 0
      for j, batch in pairs(splitted_current_triple_batch) do
         local curr_output = model:forward({batch[{{}, 1}], batch[{{}, 2}], batch[{{}, 3}]}):float()
         if j == #splitted_current_triple_batch then
            outputs[{{k*batch_size+1, outputs:size(1)}}] = curr_output
         else
            outputs[{{k*batch_size+1, (k+1)*batch_size}}] = curr_output
         end

         k = k + 1
      end

      local subj_hits, obj_hits, correct_subj_rank, correct_obj_rank = eval_triple_ranking(
                                                    current_triple_batch:float(),
                                                    num_entities, 
                                                    outputs,
                                                    inverted_relations,
                                                    topn
        )

      average_subj_hits = average_subj_hits + subj_hits
      average_obj_hits = average_obj_hits + obj_hits
      mean_subj_rank = mean_subj_rank + correct_subj_rank
      mean_obj_rank = mean_obj_rank + correct_obj_rank
      mean_reciprocal_subj_rank = mean_reciprocal_subj_rank + 1/correct_subj_rank
      mean_reciprocal_obj_rank = mean_reciprocal_obj_rank + 1/correct_obj_rank
   end

   return {
        average_subj_hits=average_subj_hits,
        average_obj_hits=average_obj_hits,
        mean_subj_rank=mean_subj_rank,
        mean_obj_rank=mean_obj_rank,
        mean_reciprocal_subj_rank=mean_reciprocal_subj_rank,
        mean_reciprocal_obj_rank=mean_reciprocal_obj_rank,
        mean_rank = (mean_subj_rank+mean_obj_rank)/2,
        mean_reciprocal_rank = (mean_reciprocal_subj_rank+mean_reciprocal_obj_rank)/2,
        hits=(average_subj_hits+average_obj_hits)/2
   }
end

local title = "### Link prediction evaluation ###"

print(title)

--[[
    JSON configuration file parameters:
        - test_filename: filename of triples data
        - triple_delimiter: delimiter of triples elements
        - model_filename: filename of learnt model
        - kb_index_filename: filename of knowledge base index
        - topn: cutoff for the link prediction task
        - batch_size: number of training examples in a batch
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

local cuda = conf_data["cuda"]

if cuda then
   require "cutorch";
   require "cunn";
   cutorch.setDevice(1)
end

print("-- Loading knowledge base index" .. conf_data["kb_index_filename"])
local kb_index = torch.load(conf_data["kb_index_filename"])

print("-- Loading triples data: " .. conf_data["test_filename"])
local triples = read_triples(conf_data["test_filename"], conf_data["triple_delimiter"], kb_index)

print("-- Loading model: " .. conf_data["model_filename"])
local model = torch.load(conf_data["model_filename"])

local topn = conf_data["topn"]
local batch_size = conf_data["batch_size"]
local num_triples = triples:size(1)

print("Dataset stats:")
print("Number of triples:\t" .. num_triples)
print("Number of entities:\t" .. #kb_index["entity2id"])
print("Number of relations:\t" .. #kb_index["relation2id"])

print(evaluate_predictions(model:get(1):get(1), triples, kb_index, topn, batch_size, cuda))
