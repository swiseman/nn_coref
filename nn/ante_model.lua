require 'nn'
require 'coref_utils'
require 'sparse_doc_data'
local mu = require 'model_utils'

torch.manualSeed(2)

do 
  local AnteModel = torch.class('AnteModel')

  function AnteModel:__init(pwD, hiddenPW, cuda, dop) 
    torch.manualSeed(2)
    if cuda then
      cutorch.manualSeed(2)
    end
    self.hiddenPW = hiddenPW
    
    local pwNet = nn.Sequential()
    pwNet:add(nn.LookupTable(pwD,hiddenPW))
    pwNet:add(nn.Sum(2))
    pwNet:add(nn.Add(hiddenPW))
    pwNet:add(nn.Tanh())
    pwNet:add(nn.Dropout(dop))
    pwNet:add(nn.Linear(hiddenPW,1))    
    
    -- make sure contiguous, and do sparse init while we're at it
    recSutsInit(pwNet,15)
    pwNet:get(1).weight[-1]:fill(0) -- assume last feature is a dummy, padding feature
    self.pwNet = cuda and pwNet:cuda() or pwNet
    collectgarbage()
  end 

  function AnteModel:docGrad(d,batch,clust,deltTensor,numMents)
    for m = 2, numMents do -- ignore first guy; always NA
      if clust:anaphoric(m) then 
        local start = ((m-2)*(m-1))/2 -- one behind first pair for mention m     
        local scores = self.pwNet:forward(batch:sub(start+1,start+m-1)):squeeze(2)
        local late = maxGoldAnt(clust,scores,m,0)
        local pred = simpleAnteLAArgmax(clust.m2c,scores,m,late,0)  
        if clust.m2c[pred] ~= clust.m2c[late] then
          self.pwNet:forward(batch:sub(start+pred,start+pred))
          self.pwNet:backward(batch:sub(start+pred,start+pred),deltTensor)
          self.pwNet:forward(batch:sub(start+late,start+late))
          self.pwNet:backward(batch:sub(start+late,start+late),-deltTensor)
        end
      end
    end
  end


  function AnteModel:getDevAcc(pwDevData,devClusts,cuda)
    assert(self.pwNet:get(1).weight[-1]:abs():sum() == 0)
    assert(self.pwNet.train == false)
    local total = 0
    local correct = 0
    for d = 1, pwDevData.numDocs do
      if d % 100 == 0 then
          print("dev doc " .. tostring(d))
          collectgarbage()
      end
      local numMents = pwDevData:numMents(d)
      local docBatch = pwDevData:getDocBatch(d)
      if cuda then
        docBatch = docBatch:cuda()
      end
      for m = 2, numMents do
        if devClusts[d]:anaphoric(m) then
          local start = ((m-2)*(m-1))/2 
          local scores = self.pwNet:forward(docBatch:sub(start+1,start+m-1)):squeeze(2)
          local _, pred = torch.max(scores,1)
          total = total + 1
          if devClusts[d].m2c[m] == devClusts[d].m2c[pred[1]] then
            correct = correct + 1
          end
        end 
      end
    end
    return correct/total
  end

  function AnteModel:docLoss(d,batch,clust,numMents)
    local loss = 0
    for m = 2, numMents do -- ignore first guy; always NA
      if clust:anaphoric(m) then
        local start = ((m-1)*(m-2))/2 -- index one behind first antecedent for this mention (in pwData)
        local scores = self.pwNet:forward(batch:sub(start+1,start+m-1)):squeeze(2)
        local late = maxGoldAnt(clust,scores,m,0)
        local pred = simpleAnteLAArgmax(clust.m2c,scores,m,late,0)
        if clust.m2c[pred] ~= clust.m2c[late] then
          loss = loss + (1 + scores[pred] - scores[late])
        end
      end
    end
    return loss
  end  
  
end

function train(pwData,clusts,pwDevData,devClusts,cuda)
  local anteModel = AnteModel(pwData.maxFeat+1, opts.H, cuda, opts.dop)
  local serFi = string.format("models/%s_%d.model", opts.savePrefix, opts.H)       
  local params, gradParams = anteModel.pwNet:getParameters()
  local optState = {}
  local deltTensor = cuda and torch.ones(1,1):cuda() or torch.ones(1,1)
  for t = 1, opts.nEpochs do
    print("epoch: " .. tostring(t))
    anteModel.pwNet:training()
    -- use document sized minibatches
    for d = 1, pwData.numDocs do
      if d % 200 == 0 then
        print("doc " .. tostring(d))
        collectgarbage()
      end 
      local batch = pwData:getDocBatch(d)
      if cuda then
        batch = batch:cuda()
      end
      gradParams:zero()
      anteModel:docGrad(d,batch,clusts[d],deltTensor,pwData:numMents(d))
      -- do pw gradients
      mu.adagradStep(params,gradParams,opts.eta,optState) 
    end

    print("evaluating on dev...")
    anteModel.pwNet:evaluate()
    local currAcc = anteModel:getDevAcc(pwDevData,devClusts,cuda)
    print("Acc " .. tostring(currAcc))
    print("")
  end
  if opts.save then
    print("overwriting params...")
    torch.save(serFi..string.format("-pw-%f",opts.eta), anteModel.pwNet)
  end  
end


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training ante model')
cmd:text()
cmd:text('Options')
cmd:option('-H', 700, 'Hidden layer size')
cmd:option('-trainClustFile', '../SMALLTrainOPCs.txt', 'Train Oracle Predicted Clustering File')
cmd:option('-devClustFile', '../SMALLDevOPCs.txt', 'Dev Oracle Predicted Clustering File')
cmd:option('-pwTrFeatPrefix', 'train_small', 'Expects train pairwise features in <pwTrFeatPfx>-pw-*.h5')
cmd:option('-pwDevFeatPrefix', 'dev_small', 'Expects dev pairwise features in <pwDevFeatPfx>-pw-*.h5')
cmd:option('-nEpochs', 20, 'Number of epochs to train')
cmd:option('-save', false, 'Save best model')
cmd:option('-savePrefix', 'small', 'Prefixes saved model with this')
cmd:option('-gpuid', -1, 'if >= 0, gives idx of gpu to use')
cmd:option('-eta', 0.1, 'adagrad learning rate')
cmd:option('-dop', 0.5, 'dropout rate')
cmd:text()

-- Parse input options
opts = cmd:parse(arg)

if opts.gpuid >= 0 then
  print('using cuda on gpu ' .. opts.gpuid)
  require 'cutorch'
  require 'cunn'
  cutorch.manualSeed(2)
  cutorch.setDevice(opts.gpuid+1)
end

function main()
  local pwTrData = SpDMPWData.loadFromH5(opts.pwTrFeatPrefix)
  print("read pw train data")
  print("max pw feature is: " .. pwTrData.maxFeat)
  local pwDevData = SpDMPWData.loadFromH5(opts.pwDevFeatPrefix)
  print("read pw dev data")
  local trClusts = getOPCs(opts.trainClustFile,pwTrData)
  print("read train clusters")
  local devClusts = getOPCs(opts.devClustFile,pwDevData)
  print("read dev clusters") 
 
  train(pwTrData,trClusts,pwDevData,devClusts,opts.gpuid >= 0)
end

main()
