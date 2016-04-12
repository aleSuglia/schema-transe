require "dataset";
require "tds";

function eval_triple_ranking(triple_batch, outputs, relation_index, topn)
   local y, indexes = torch.topk(outputs, topn)
   local hits = 0
   local current_relation = triple_batch[1][2]
   local current_relation_pairs = relation_index[current_relation]
   
   for i=1, indexes:size(1) do
      for _, v in pairs(current_relation_pairs) do
         if v == triple_batch[i][1] or v == triple_batch[i][3] then
            hits = hits + 1
            break
         end
      end
   end

   return hits/topn
end

function build_relation_index(triples_data)
   local triples = triples_data["triples"]
   local relation_index = tds.Hash()
   
   for i=1, triples:size(1) do
      local current_triple = triples[i]
      local current_relation = current_triple[2]
      if not relation_index[current_relation] then
         relation_index[current_relation] = tds.Vec()
      end

      relation_index[current_relation]:insert({current_triple[1], current_triple[3]})
   end

   return relation_index
end

function evaluate_predictions(model, triples_data, topn)
   local test_set = triples_data["triples"]
   local num_entities = #triples_data["entity2id"]
   local average_hits = 0
   local relation_index = build_relation_index(triples_data)
   
   for i=1, test_set:size(1) do
      local current_triple_batch = torch.Tensor(2*num_entities, 3)
      local current_triple = test_set[i]
      local entities_ids_nosubj = torch.Tensor(num_entities-1)
      local entities_ids_noobj = torch.Tensor(num_entities-1)
      
      -- Gets entities ids
      local key_index_nosubj = 1      
      local key_index_noobj = 1
      for k, _ in pairs(triples_data["id2entity"]) do
         if k ~= current_triple[1] then
            entities_ids_nosubj[key_index_nosubj] = k
            key_index_nosubj = key_index_nosubj + 1
         end

         if k ~= current_triple[3] then
            entities_ids_noobj[key_index_noobj] = k
            key_index_noobj = key_index_noobj + 1
         end
      end
      
      current_triple_batch[{1, {}}] = current_triple

      -- Initialize relation and object columns in the range (2, num_entities) using
      -- current triple relation and object ids
      local rel_obj_columns = current_triple_batch:narrow(2, 2, 2)
      local rel_obj_rows = rel_obj_columns:narrow(1, 2, num_entities)
      rel_obj_rows:narrow(2, 1, 1):fill(current_triple[2])
      rel_obj_rows:narrow(2, 2, 1):fill(current_triple[3])

      -- Initialize the subject column values in the range (2, num_entities) using corrupted subject id 
      current_triple_batch[{{2, num_entities}, 1}] = entities_ids_nosubj

      -- Initialize subject and relation columns in the range (num_entities+1, 2*num_entities) using
      -- current triple subject and relation ids
      local subj_rel_columns = current_triple_batch:narrow(2, 1, 2)
      local subj_rel_rows = subj_rel_columns:narrow(1, num_entities+1, num_entities)  
      subj_rel_rows:narrow(2, 1, 1):fill(current_triple[1])
      subj_rel_rows:narrow(2, 2, 1):fill(current_triple[2])

      -- Initialize the subject column values in the range (num_entities+1, 2*num_entities) using corrupted object id
      current_triple_batch[{{num_entities+1, 2*num_entities-1}, 3}] = entities_ids_noobj

      local splitted_current_triple_batch = current_triple_batch:split(batch_size)
      local outputs = torch.Tensor(current_triple_batch:size(1))

      local i = 0
      for _, batch in pairs(splitted_current_triple_batch) do
         outputs[{{i*batch_size+1, (i+1)*batch_size}}] = model:forward(batch)
         i = i + 1
      end
            
      average_hits = average_hits + eval_triple_ranking(current_triple_batch, outputs, relation_index, topn)
   end

   return average_hits/num_entities 
end
