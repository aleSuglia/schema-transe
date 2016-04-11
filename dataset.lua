local tds = require "tds";
local stringx = require "pl.stringx";
require "math";

function read_triples_data(triples_filename, delimiter)
    local num_lines = 0
   
    for line in io.lines(triples_filename) do
        num_lines = num_lines + 1
    end

    local delimiter = delimiter or ","
    local triples = torch.Tensor(num_lines, 3)
    local entity2id = tds.Hash()
    local id2entity = tds.Hash()
    local relation2id = tds.Hash()
    local id2relation = tds.Hash()
    local num_entities = 1
    local num_relations = 1
    local i = 1
    
    for line in io.lines(triples_filename) do
        local splitted_line = stringx.split(line, delimiter)

        local subject = splitted_line[1]
        local property = splitted_line[2]
        local object = splitted_line[3]

        if not entity2id[subject] then
           entity2id[subject] = num_entities
           id2entity[num_entities] = subject
           num_entities = num_entities + 1
        end

        if not relation2id[property] then
           relation2id[property] = num_relations
           id2relation[num_relations] = property
           num_relations = num_relations + 1
        end

        if not entity2id[object] then
           entity2id[object] = num_entities           
           id2entity[num_entities] = object
           num_entities = num_entities + 1
        end

        triples[i][1] = entity2id[subject]
        triples[i][2] = relation2id[property]
        triples[i][3] = entity2id[object]

        i = i + 1
    end

    return {
        triples = triples,
        entity2id = entity2id,
        id2entity = id2entity,
        relation2id = relation2id,
        id2relation = id2relation
    }
end

function sample_corrupted_triple(triples_data, triple)
   local id2entity = triples_data["id2entity"]
   local entity2id = triples_data["entity2id"]
   local random_entity = entity2id[id2entity[math.random(1, #id2entity)]]
   local corrupted_triple
   
   -- Decides if the subject or the object should be corrupted 
   if math.random() < 0.5 then
      corrupted_triple = {random_entity, triple[2], triple[3]}
   else
      corrupted_triple = {triple[1], triple[2], random_entity}
   end

   return torch.Tensor(corrupted_triple)
end
