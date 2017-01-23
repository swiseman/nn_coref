require 'nn'
require 'coref_utils'
require 'sparse_doc_data'
local mu = require 'model_utils'

ANAModel = {}
ANAModel.__index = ANAModel

function ANAModel.make(naD, hiddenUnary, c0, c1, dop) 
  torch.manualSeed(2)
    model = {}
    setmetatable(model,ANAModel)
    model.hiddenUnary = hiddenUnary
    model.c0 = c0
    model.c1 = c1

    local naNet = nn.Sequential()
    naNet:add(nn.LookupTable(naD,hiddenUnary))
    naNet:add(nn.Sum(2))
    naNet:add(nn.Add(hiddenUnary))
    naNet:add(nn.Tanh())
    naNet:add(nn.Dropout(dop))
    naNet:add(nn.Linear(hiddenUnary,1))
    model.naNet = naNet
    
    -- make sure contiguous, and do sparse init while we're at it
    recSutsInit(naNet,15) 
    model.naNet:get(1).weight[-1]:fill(0)
    
    collectgarbage()
    return model
end 


-- gets the gradient of loss wrt score for each datapt, 
-- based on a slack rescaled (binary) hinge loss
function ANAModel:srHingeBackwd(forwardScores, targets)
   -- presumably faster than doing more vectorized stuff
   local delt = torch.Tensor(targets:size(1))
   local delt_data = torch.data(delt)
   local target_data = torch.data(targets)
   local scores_data = torch.data(forwardScores)
   for i = 0, targets:size(1)-1 do
       if target_data[i] == 1 and scores_data[i] < 1 then
           delt_data[i] = -self.c1
       elseif target_data[i] == -1 and scores_data[i] > -1 then
           delt_data[i] = self.c0
       else
           delt_data[i] = 0
       end
   end
   return delt
end


-- computes gradients assuming parameters have been zeroed out
-- naData is an SpDMData   
function ANAModel:miniBatchGrad(d,naData,targets)
    -- use doc-sized mini batches
    local docMB = naData:docBatch(d)
    docMB = docMB:sub(2,docMB:size(1))
    local scores = self.naNet:forward(docMB)
    local delts = self:srHingeBackwd(scores,targets[d]:sub(2,targets[d]:size(1)))
    -- need to make delts 2d to push backward
    self.naNet:backward(docMB,delts:view(delts:size(1),1))
end

function ANAModel:getDevF(naDevData,devTargets)
  assert(self.naNet.train == false)
  assert(self.naNet:get(1).weight[-1]:abs():sum() == 0) 
    local tp,tn,fp,fn = 0,0,0,0
    for d = 1, naDevData.numDocs do
      if d % 100 == 0 then
        print("dev doc " .. tostring(d))
        collectgarbage()
      end
      local docMB = naDevData:docBatch(d)
      docMB = docMB:sub(2,docMB:size(1))
      local scores = self.naNet:forward(docMB)
      local targets = devTargets[d]:sub(2,devTargets[d]:size(1))
      for m = 1, scores:size(1) do
        if targets[m] == 1 then
          if scores[m][1] >= 0 then
            tp = tp + 1
          else
            fn = fn + 1
          end
        else
          if scores[m][1] < 0 then
            tn = tn + 1
          else
            fp = fp + 1
          end
        end
      end
    end
    local prec = tp/(tp + fp)
    local rec = tp/(tp + fn)
    local F = 2*prec*rec/(prec+rec)
    return prec,rec,F
end

function ANAModel:docLoss(d,naData,targets)
    local loss = 0
    local docMB = naData:docMiniBatch(d)
    local scores = self.naNet:forward(docMB)
    local target_data = torch.data(targets[d])
    local scores_data = torch.data(scores)
    for i = 0, targets[d]:size(1)-1 do
       if target_data[i] == 1 and scores_data[i] < 1 then
           loss = loss + self.c1*(1-scores_data[i])
       elseif target_data[i] == -1 and scores_data[i] > -1 then
           loss = loss + self.c0*(1 + scores_data[i])
       end
    end    
    return loss
end

function makeTargets(naSpData,clusts) -- note different than in original file
  local targets = {}
  for i = 1, naSpData.numDocs do
    local t = torch.ones(naSpData:numMents(i))
    for j = 1, naSpData:numMents(i) do
      if not clusts[i]:anaphoric(j) then
        t[j] = -1
      end
    end
    assert(t[1] == -1)
    table.insert(targets,t)
  end
  return targets   
end


function train(naTrData,trTargets,naDevData,devTargets)
  local serFi = string.format("models/%s_%d.model", opts.savePrefix, opts.H)
  local anaModel = ANAModel.make(naTrData.maxFeat+1, opts.H, opts.c0, opts.c1, opts.dop)   
  local params, gradParams = anaModel.naNet:getParameters()
  local optState = {}
  local bestF = 0

  for n = 1, opts.nEpochs do
    anaModel.naNet:training()
    print("epoch: " .. tostring(n))
    -- use document sized minibatches
    for d = 1, naTrData.numDocs do
      if d % 200 == 0 then
        print("doc " .. tostring(d))
        collectgarbage()
      end 

      gradParams:zero()
      anaModel:miniBatchGrad(d,naTrData,trTargets)
      mu.adagradStep(params,gradParams,opts.eta,optState) 
    end

    print("evaluating on dev...")
    anaModel.naNet:evaluate()
    local prec,rec,currF = anaModel:getDevF(naDevData,devTargets)
  end

  if opts.save then
    print("overwriting params...")
    torch.save(serFi .. string.format("-na-%f",opts.eta), anaModel.naNet)
  end

end



cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training anaph model')
cmd:text()
cmd:text('Options')
cmd:option('-H', 200, 'Hidden layer size')
cmd:option('-trainClustFile', '../SMALLTrainOPCs.txt', 'Train Oracle Predicted Clustering File')
cmd:option('-devClustFile', '../SMALLDevOPCs.txt', 'Dev Oracle Predicted Clustering File')
cmd:option('-anaTrFeatPrefix', 'train_small', 'Expects train anaphoricity features in <anaTrFeatPfx>-na-*.h5')
cmd:option('-anaDevFeatPrefix', 'dev_small', 'Expects dev anaphoricity features in <anaDevFeatPfx>-na-*.h5')
cmd:option('-c0', 1, 'False positive cost')
cmd:option('-c1', 1.35, 'False negative cost')
cmd:option('-nEpochs', 9, 'Number of epochs to train')
cmd:option('-save', false, 'Save best model')
cmd:option('-savePrefix', 'simple', 'Prefixes saved model with this')
cmd:option('-t', 2, "Number of threads")
cmd:option('-eta', 0.1, 'Adagrad learning rate')
cmd:option('-dop', 0, 'dropout rate')
cmd:text()

-- Parse input options
opts = cmd:parse(arg)

if opts.t > 0 then
    torch.setnumthreads(opts.t)
end
print("Using " .. tostring(torch.getnumthreads()) .. " threads")


function main()
  local anaTrData = SpKFDMData(opts.anaTrFeatPrefix)
  print("read anaph train data")
  print("max ana feature is: " .. anaTrData.maxFeat)
  local anaDevData = SpKFDMData(opts.anaDevFeatPrefix)
  print("read anaph dev data") 
  local trClusts = getOPCs(opts.trainClustFile,anaTrData)
  print("read train clusters")
  local devClusts = getOPCs(opts.devClustFile,anaDevData)
  print("read dev clusters")
  -- make targets for training     
  local trTargets = makeTargets(anaTrData,trClusts)
  local devTargets = makeTargets(anaDevData,devClusts)
  train(anaTrData,trTargets,anaDevData,devTargets)

end

main()
