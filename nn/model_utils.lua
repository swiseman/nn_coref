local model_utils = {}

function model_utils.adagradStep(x,dfdx,eta,state)
  if not state.var then
    state.var = torch.Tensor():typeAs(x):resizeAs(x):zero()
    state.std = torch.Tensor():typeAs(x):resizeAs(x)
  end
  state.var:addcmul(1,dfdx,dfdx)
  state.std:sqrt(state.var)
  x:addcdiv(-eta, dfdx, state.std:add(1e-10))
end

function model_utils.make_sp_mlp(D,H,zeroLast,justFirstLayer,dop)
  local mlp = nn.Sequential()
  mlp:add(nn.LookupTable(D,H))   
  mlp:add(nn.Sum(2))
  mlp:add(nn.Add(H)) -- add a bias
  mlp:add(nn.Tanh())
  if not justFirstLayer then
    mlp:add(nn.Dropout(dop or 0.5))
    mlp:add(nn.Linear(H,1))
  end
  -- make sure contiguous, and do sparse sutskever init while we're at it
  recSutsInit(mlp,15)
  if zeroLast then
    mlp:get(1).weight[-1]:fill(0)
  end
  return mlp
end


function model_utils.make_sp_and_dense_mlp(spD,dD,H,zeroLast,justFirstLayer,dop)
  local mlp = nn.Sequential()
  local parLayer = nn.ParallelTable()
  local left = nn.Sequential()
  left:add(nn.LookupTable(spD,H))
  left:add(nn.Sum(2)) -- after this pt, will have totalNumMents x H
  local right = nn.Sequential()
  right:add(nn.Linear(dD,H)) -- just handles the distance feature (and the bias, conveniently)
  parLayer:add(left)
  parLayer:add(right)
  mlp:add(parLayer)
  mlp:add(nn.CAddTable())
  mlp:add(nn.Tanh())
  if not justFirstLayer then
    mlp:add(nn.Dropout(dop or 0.5))
    mlp:add(nn.Linear(H,1))
  end
  recSutsInit(mlp,15)
  if zeroLast then
    mlp:get(1):get(1):get(1).weight[-1]:fill(0)
  end
  return mlp
end


function sparseSutsMatInit(W,numNZ,scale)
  local numNZ = numNZ or 15
  local scale = scale or 0.25
  local m = W:size(1)
  local n = W:size(2)
  -- zero everything out
  W:fill(0)
  if n >= m then -- assume columns are features and rows are hidden dims
    numNZ = math.min(numNZ,n)
    for i = 1, m do
      local perm = torch.randperm(n)
      -- probably better ways of doing this
      local r = torch.randn(numNZ)*scale
      for j = 1, numNZ do
        W[i][perm[j]] = r[j]
      end
    end
  else -- assume rows are features and columns hidden dims
    numNZ = math.min(numNZ,m)
    for j = 1, n do
      local perm = torch.randperm(m)
      local r = torch.randn(numNZ)*scale
      for i = 1, numNZ do
        W[perm[i]][j] = r[i]
      end
    end
  end
end

function recSutsInit(net,numNZ) -- assuming no module can have weight and children
  local numNZ = numNZ or 15
  if net.weight and net.bias then
    sparseSutsMatInit(net.weight, math.min(numNZ,net.weight:size(1),net.weight:size(2)))
    net.bias:fill(0.5)
  elseif net.weight then
    sparseSutsMatInit(net.weight, math.min(numNZ,net.weight:size(1),net.weight:size(2)))
  elseif net.bias then
    net.bias:fill(0.5)
  elseif net.modules and #net.modules > 0 then
    for layer, subnet in ipairs(net.modules) do
      recSutsInit(subnet, numNZ)
    end
  end
end

-- stolen from https://github.com/karpathy/char-rnn/blob/master/util/model_utils.lua
function model_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
      local tn = torch.typename(layer)
	    local net_params, net_grads = networks[i]:parameters()
	    if net_params then
	      for _, p in pairs(net_params) do
		      parameters[#parameters + 1] = p
	      end
	      for _, g in pairs(net_grads) do
		      gradParameters[#gradParameters + 1] = g
	      end
	    end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

  

return model_utils
