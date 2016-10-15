local tds = require "tds";
local stringx = require "pl.stringx";

-- Builds a dictionary-like data structure wich maps URIs to ids 
function build_kb_index(train_filename, test_filename, validation_filename, delimiter)
    local delimiter = delimiter or ","
    local entity2id = tds.Hash()
    local id2entity = tds.Hash()
    local relation2id = tds.Hash()
    local id2relation = tds.Hash()
    local num_entities = 1
    local num_relations = 1

    local function build_index(filename)
        for line in io.lines(filename) do
            local splitted_line = stringx.split(line, delimiter)
            local subject = tonumber(splitted_line[1])
            local property = tonumber(splitted_line[2])
            local object = tonumber(splitted_line[3])

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
        end
    end
    
    build_index(train_filename)
    build_index(test_filename)
    build_index(validation_filename)

    return {
        entity2id = entity2id,
        id2entity = id2entity,
        relation2id = relation2id,
        id2relation = id2relation
    }
end

-- Reads triples (one for each line) from from a file where each element is separated by the specified delimiter 
function read_triples(triples_filename, delimiter, kb_index)
    local num_lines = 0

    for _ in io.lines(triples_filename) do
        num_lines = num_lines + 1
    end

    local delimiter = delimiter or ","
    local triples = torch.Tensor(num_lines, 3)
    local entity2id = kb_index["entity2id"]
    local relation2id = kb_index["relation2id"]
    local i = 1

    for line in io.lines(triples_filename) do
        local splitted_line = stringx.split(line, delimiter)
        local subject = tonumber(splitted_line[1])
        local property = tonumber(splitted_line[2])
        local object = tonumber(splitted_line[3])

        triples[i][1] = entity2id[subject]
        triples[i][2] = relation2id[property]
        triples[i][3] = entity2id[object]

        i = i + 1
    end

    return triples
end
