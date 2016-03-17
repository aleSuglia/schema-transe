require "nn";

function build_transe_model(entity_embeddings_size, relation_embeddings_size)
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

   local full_model = nn.Sequential()

   full_model:add(nn.ParallelTable())

   return full_model
   
end

batch_size = 2

correct_triples_batches = {
   {
      {1, 5, 7},
      {1, 5, 7},
      {1, 5, 7}

   },

   {
      {1, 10, 3},
      {1, 10, 3},
      {1, 10, 3}
   }

}

corrupted_triples_batches = {
   {
      {1, 10, 3},
      {3, 5, 7},
      {1, 5, 3}

   },

   {
      {1, 10, 3},
      {7, 10, 3},
      {1, 10, 7}
   }

}


target_batches = {
   {
      {1},
      {-1},
      {-1}
   },

   {
      {1},
      {-1},
      {-1}
   }
}


correct_triples_batches = torch.Tensor(correct_triples_batches)
triples_batches = torch.Tensor(triples_batches)
target_batches = torch.Tensor(target_batches)

print(triples_batches:size())
--print(target_batches)

model = build_transe_model(0, 0)

print(model:forward(triples_batches))
