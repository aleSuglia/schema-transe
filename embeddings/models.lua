require "nn";

function normalize_lookup_table(lookup_table, p)
    local norms = torch.norm(lookup_table.weight, p, 2)
    for i=1,norms:size(1) do
        lookup_table.weight[{i, {}}]:div(norms[i][1])
    end
end

function build_transe_model(num_entities, num_relations, embeddings_size)
    local function init_lookup_weights(entities_lookup, relations_lookup, embeddings_size)
        entities_lookup.weight:uniform(-6/math.sqrt(embeddings_size), 6/math.sqrt(embeddings_size))
        relations_lookup.weight:uniform(-6/math.sqrt(embeddings_size), 6/math.sqrt(embeddings_size))
        normalize_lookup_table(relations_lookup, 2)
    end

    local full_model = nn.Sequential()
    local entities_lookup = nn.LookupTable(num_entities, embeddings_size)
    local relations_lookup = nn.LookupTable(num_relations, embeddings_size)
    init_lookup_weights(entities_lookup, relations_lookup, embeddings_size)

    local correct_triples_model = nn.Sequential()
    local correct_parallel_triples = nn.ParallelTable()
    correct_parallel_triples:add(entities_lookup)
    correct_parallel_triples:add(relations_lookup)
    correct_parallel_triples:add(entities_lookup:clone("weight", "bias", "gradWeight", "gradBias"))
    correct_triples_model:add(correct_parallel_triples)
    local correct_concat_triples = nn.ConcatTable()
        :add(nn:Identity())
        :add(nn.Identity())
    correct_triples_model:add(correct_concat_triples)
    local correct_add_triples = nn.ParallelTable()
        :add(nn.Sequential()
        :add(nn.NarrowTable(1, 2))
        :add(nn.CAddTable()))
        :add(nn.SelectTable(3))
    correct_triples_model:add(correct_add_triples)
    correct_triples_model:add(nn.PairwiseDistance(1))

    local corrupted_triples_model = nn.Sequential()
    local corrupted_parallel_triples = nn.ParallelTable()
    corrupted_parallel_triples:add(entities_lookup:clone("weight", "bias", "gradWeight", "gradBias"))
    corrupted_parallel_triples:add(relations_lookup:clone("weight", "bias", "gradWeight", "gradBias"))
    corrupted_parallel_triples:add(entities_lookup:clone("weight", "bias", "gradWeight", "gradBias"))
    corrupted_triples_model:add(corrupted_parallel_triples)
    local corrupted_concat_triples = nn.ConcatTable()
        :add(nn:Identity())
        :add(nn.Identity())
    corrupted_triples_model:add(corrupted_concat_triples)
    local corrupted_add_triples = nn.ParallelTable()
        :add(nn.Sequential()
        :add(nn.NarrowTable(1, 2))
        :add(nn.CAddTable()))
        :add(nn.SelectTable(3))
    corrupted_triples_model:add(corrupted_add_triples)
    corrupted_triples_model:add(nn.PairwiseDistance(1))

    full_model:add(nn.ParallelTable()
        :add(correct_triples_model)
        :add(corrupted_triples_model))

    return full_model
end
