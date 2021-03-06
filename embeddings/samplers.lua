local cjson = require "cjson";
local http = require "socket.http";
local ltn12 = require "ltn12";

-- Performs a HTTP POST
local function post(request_data, server_url)
    local json_request_data = cjson.encode(request_data)
    local source = ltn12.source.string(json_request_data)
    local response = {}
    -- Save response from server in chunks
    local sink = ltn12.sink.table(response)
    local ok, code, headers = http.request {
        url = server_url,
        method = "POST",
        headers = {
            ["Content-Type"] = "application/json",
            ["Content-Length"] = #json_request_data
        },
        source = source,
        sink = sink
    }

    return cjson.decode(table.concat(response))
end

-- Samples a random triple according to a uniform probability distribution over the dataset triples
function sample_random(triple_batch, kb_index, params)
    local num_corrupted = params["size"]

    local function sample_corrupted_triple(triple)
        local id2entity = kb_index["id2entity"]
        local entity2id = kb_index["entity2id"]

        local corrupted_triples = torch.Tensor(num_corrupted, 3)

        for i = 1, num_corrupted do
            local random_entity = entity2id[id2entity[math.random(1, #id2entity)]]

            -- Decides if the subject or the object should be corrupted 
            if math.random() < 0.5 then
                while random_entity == triple[1] do
                    random_entity = entity2id[id2entity[math.random(1, #id2entity)]]
                end
                corrupted_triples[{ i, {} }] = torch.Tensor { random_entity, triple[2], triple[3] }
            else
                while random_entity == triple[3] do
                    random_entity = entity2id[id2entity[math.random(1, #id2entity)]]
                end
                corrupted_triples[{ i, {} }] = torch.Tensor { triple[1], triple[2], random_entity }
            end
        end
        return corrupted_triples
    end

    local corrupted_batch = torch.Tensor(triple_batch:size(1) * num_corrupted, 3)

    for i = 1, triple_batch:size(1) do
        local correct_triple = triple_batch[i]
        local corrupted_triples = sample_corrupted_triple(correct_triple)
        corrupted_batch[{ { (i - 1) * num_corrupted + 1, i * num_corrupted }, {} }] = corrupted_triples
    end

    return corrupted_batch
end

-- Samples a corrupted triple using a schema-aware reasoner
function sample_reasoner(triple_batch, kb_index, params)
    local request_data = {}
    local id2entity = kb_index["id2entity"]
    local id2relation = kb_index["id2relation"]
    local entity2id = kb_index["entity2id"]
    local relation2id = kb_index["relation2id"]

    request_data["triples"] = {}
    for i = 1, triple_batch:size(1) do
        request_data["triples"][i] = {
            subject = id2entity[triple_batch[i][1]],
            predicate = id2relation[triple_batch[i][2]],
            object = id2entity[triple_batch[i][3]]
        }
    end
    request_data["size"] = params["size"]

    local corrupted_triples = post(request_data, params["server_url"])
    local corrupted_triple_batch = torch.Tensor(triple_batch:size(1) * params["size"], 3)

    local index = 1
    for _, cts in pairs(corrupted_triples) do
        for _, ct in pairs(cts) do
            corrupted_triple_batch[{ index, {} }] = torch.Tensor {
                entity2id[tonumber(ct["subject"])],
                relation2id[tonumber(ct["predicate"])],
                entity2id[tonumber(ct["object"])]
            }
            index = index + 1
        end
    end

    return corrupted_triple_batch
end

-- Returns the sampler with the specified id
function create_sampler(sampler_id)
    if sampler_id == "random" then
        return sample_random
    elseif sampler_id == "reasoner" then
        return sample_reasoner
    end
    error("Invalid sampler ID!")
end
