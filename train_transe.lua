require "dataset.lua";

local title = "### TransE ###"

print(title)

local cmd = torch.CmdLine()
cmd:text()
cmd:text(title)
cmd:text()
cmd:text("Options:")
cmd:option("-train", "", "Training set path")
cmd:text()

local params = cmd:parse(arg)

local triples_data = read_triples(params.train)

print(triples_data["triples"]:size())
