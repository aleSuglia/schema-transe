require "nn";

-- Builds TransE model by Bordes et. al
function build_transe_model(triples_data, entities_embeddings_size, relations_embeddings_size)
   local num_entities = #triples_data["entity2id"]
   local num_relations = #triples_data["relation2id"]
   --local num_entities = 7
   --local num_relations = 10
      
   local full_model = nn.Sequential()
   local entities_lookup = nn.Sequential():add(nn.LookupTable(num_entities, entities_embeddings_size)):add(nn.Normalize(2))
   local relations_lookup = nn.LookupTable(num_relations, relations_embeddings_size)

   local correct_sp = nn.SplitTable(2)
   correct_sp.updateGradInput = function() end
   local correct_triples_model = nn.Sequential():add(correct_sp)
   local correct_parallel_triples = nn.ParallelTable()
   correct_parallel_triples:add(entities_lookup)
   correct_parallel_triples:add(relations_lookup)
   correct_parallel_triples:add(entities_lookup:clone("weight", "bias"))
   correct_triples_model:add(correct_parallel_triples)
   local correct_concat_triples = nn.ConcatTable()
      :add(nn:Identity())
      :add(nn.Identity())
   correct_triples_model:add(correct_concat_triples)
   local correct_add_triples = nn.ParallelTable()
      :add(nn.Sequential()
              :add(nn.NarrowTable(1,2))
              :add(nn.CAddTable()))
      :add(nn.SelectTable(3))
   correct_triples_model:add(correct_add_triples)
   correct_triples_model:add(nn.PairwiseDistance(1))

   local corrupted_sp = nn.SplitTable(2)
   corrupted_sp.updateGradInput = function() end
   local corrupted_triples_model = nn.Sequential():add(corrupted_sp)
   local corrupted_parallel_triples = nn.ParallelTable()
   corrupted_parallel_triples:add(entities_lookup:clone("weight", "bias"))
   corrupted_parallel_triples:add(relations_lookup:clone("weight", "bias"))
   corrupted_parallel_triples:add(entities_lookup:clone("weight", "bias"))
   corrupted_triples_model:add(corrupted_parallel_triples)

   local corrupted_concat_triples = nn.ConcatTable()
      :add(nn:Identity())
      :add(nn.Identity())
   corrupted_triples_model:add(corrupted_concat_triples)
   local corrupted_add_triples = nn.ParallelTable()
      :add(nn.Sequential()
              :add(nn.NarrowTable(1,2))
              :add(nn.CAddTable()))
      :add(nn.SelectTable(3))
   corrupted_triples_model:add(corrupted_add_triples)
   corrupted_triples_model:add(nn.PairwiseDistance(1))
   full_model:add(nn.ParallelTable()
                     :add(correct_triples_model)
                     :add(corrupted_triples_model))
   full_model:add(nn.CSubTable())
   
   return full_model

end

--[[
correct_triples_batches = {
   {1, 5, 7},
   {1, 5, 7},
   {1, 10, 3},
   {1, 10, 3}
}

corrupted_triples_batches = {
   {1, 5, 7},
   {6, 5, 2},
   {1, 10, 3},
   {7, 10, 5}
}

target_batches = {
   1,
   -1,
   1,
   -1
}

model = build_transe_model(nil, 10, 10)
inputs = {torch.Tensor(correct_triples_batches),
          torch.Tensor(corrupted_triples_batches)}
targets = torch.Tensor(target_batches)
outputs = model:forward(inputs)
print(outputs)

criterion = nn.HingeEmbeddingCriterion(1)

local f = criterion:forward(outputs, targets)
local df_do = criterion:backward(outputs, targets)
model:backward(inputs, df_do)
         
--]]
