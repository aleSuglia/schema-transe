file = require "pl.file";
cjson = require "cjson";
require "dataset.lua";

local title = "### Knowledge base index builder ###"

print(title)

local cmd = torch.CmdLine()
cmd:text()
cmd:text(title)
cmd:text()
cmd:text("Options:")
cmd:option("-config", "", "Filename of JSON training parameters")
cmd:text()

local params = cmd:parse(arg)
local conf_data = cjson.decode(file.read(params.config))

local train_filename = conf_data["train_filename"]
local test_filename = conf_data["test_filename"]
local validation_filename = conf_data["validation_filename"]
local triple_delimiter = conf_data["triple_delimiter"]
local kb_index_filename = conf_data["kb_index_filename"]

print("-- Building knowledge base index")
kb_index = build_kb_index(train_filename, test_filename, validation_filename, triple_delimiter)

print("-- Saving knowledge base index")
torch.save(kb_index_filename, kb_index)
