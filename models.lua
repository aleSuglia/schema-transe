require "nn";

function build_transe_model(num_entities, num_relations, entities_embeddings_size, relations_embeddings_size)
   --[[ 

      entities = {1,3,7}
      relations = {5, 10, 12}
      batch_size = 2
      num_corrupted_examples = 2 (2-dimension = 1 + num_corrupted_examples)
   
      We assume that the input has the following form:
   
      -- Input batch
      (1,.,.) = 
      1   5   7   1  10   3
      1   5   7   3   5   7
      1   5   7   1   5   3

      (2,.,.) = 
      1  10   3   1  10   3
      1  10   3   7  10   3
      1  10   3   1  10   7
      [torch.DoubleTensor of size 2x3x6]

      -- Target batch
      (1,.,.) = 
       1
      -1
      -1

      (2,.,.) = 
       1
      -1
      -1
      [torch.DoubleTensor of size 2x3x1]

   --]]
  
   local full_model = nn.ParallelTable()
   local entities_lookup = nn.LookupTable(num_entities, entities_embeddings_size)
   local relations_lookup = nn.LookupTable(num_relations, relations_embeddings_size)
   
   local correct_triples_model = nn.Sequential():add(nn.SplitTable(2))
   local correct_parallel_triples = nn.ParallelTable()
   correct_parallel_triples:add(entities_lookup)
   correct_parallel_triples:add(relations_lookup)
   correct_parallel_triples:add(entities_lookup)
   correct_triples_model:add(correct_parallel_triples)
   
   local corrupted_triples_model = nn.Sequential():add(nn.SplitTable(2))

   local corrupted_parallel_triples = nn.ParallelTable()
   corrupted_parallel_triples:add(entities_lookup)
   corrupted_parallel_triples:add(relations_lookup)
   corrupted_parallel_triples:add(entities_lookup)
   corrupted_triples_model:add(corrupted_parallel_triples)

   full_model:add(correct_triples_model)
   full_model:add(corrupted_triples_model)
   
   return full_model
end

batch_size = 6

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
   {1},
   {-1},
   {1},
   {-1}
}

correct_triples_batches = torch.Tensor(correct_triples_batches)
corrupted_triples_batches = torch.Tensor(corrupted_triples_batches)
target_batches = torch.Tensor(target_batches)

--print(triples_batches)
--print(target_batches)

model = build_transe_model(7, 10, 5, 5)

print(model:forward({correct_triples_batches, corrupted_triples_batches}))
