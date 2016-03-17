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
end
