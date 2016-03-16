local stringx = require "pl.stringx";
local file = require "pl.file";
local tds = require "tds";
local data = require "pl.data";

function read_triples(triples_filename, delimiter)
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
    local num_entities = 0
    local num_relations = 0
    local i = 1
    
    for line in io.lines(triples_filename) do
        local splitted_line = stringx.split(line, delimiter)

        local subject = splitted_line[1]
        local property = splitted_line[2]
        local object = splitted_line[3]

        if not entity2id[subject] then
            entity2id[subject] = num_entities
            num_entities = num_entities + 1
        end

        if not relation2id[property] then
            relation2id[property] = num_relations
            num_relations = num_relations + 1
        end

        if not entity2id[object] then
            entity2id[object] = num_entities
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
