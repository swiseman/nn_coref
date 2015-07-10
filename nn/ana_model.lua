require 'nn'
dofile('init.lua')
require 'cmd_adagrad'
require 'coref_utils'
require 'sparse_doc_data'

ANAModel = {}
ANAModel.__index = ANAModel

function ANAModel.make(naD, hiddenUnary, c0, c1) 
    model = {}
    setmetatable(model,ANAModel)
    model.hiddenUnary = hiddenUnary
    model.c0 = c0
    model.c1 = c1

    local naNet = nn.Sequential()
    naNet:add(nn.LookupTable(naD,hiddenUnary))
    -- lookup creates a minibatch_size x numFeats x hiddenUnary tensor; we sum to get an mv product    
    naNet:add(nn.Sum(2))
    naNet:add(nn.Add(hiddenUnary)) -- add a bias
    naNet:add(nn.Tanh())
    naNet:add(nn.Linear(hiddenUnary,1))
    model.naNet = naNet
    
    -- make sure contiguous, and do sparse Sutskever init while we're at it
    checkContigAndSutsInit(naNet,15)  
    
    model.nastates = {{},{},{},{}}
    
    collectgarbage()
    return model
end 


function ANAModel.load(serfi, c0, c1) 
    model = {}
    setmetatable(model,ANAModel)
    model.c0 = c0
    model.c1 = c1

    local naNet = torch.load(serfi)
    model.hiddenUnary = naNet:get(1).weight:size(2)
    model.naNet = naNet
    
    model.nastates = {{},{},{},{}}
    
    collectgarbage()
    return model
end 

-- this just gets the delta for each datapt based on a slack rescaled (binary) hinge loss
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


--[[ 
   - computes gradients assuming parameters have been zeroed out
   - naData is an SpDMData
--]]    
function ANAModel:docGrad(d,naData,targets)
    local docMB = naData:docMiniBatch(d)
    local scores = self.naNet:forward(docMB)
    local delts = self:srHingeBackwd(scores,targets[d])
    -- need to make delts 2d to push backward
    self.naNet:backward(docMB,delts:view(targets[d]:size(1),1))
end


function ANAModel:train(naTrData,trTargets,naDevData,devTargets,eta,lamb,
                    nEpochs,save,savePfx)
    local nEpochs = nEpochs or 5
    local serFi = string.format("models/%s_%d.model", savePfx, self.hiddenUnary)
                         
    -- reset everything
    torch.manualSeed(2)
    self.nastates = {{},{},{},{}}
    collectgarbage()           
    checkContigAndSutsInit(self.naNet,15)
    collectgarbage()
    local bestF = 0
    
    for n = 1, nEpochs do
       print("epoch: " .. tostring(n))
       -- use document sized minibatches
       for d = 1, naTrData.numDocs do
          if d % 200 == 0 then
             print("doc " .. tostring(d))
             collectgarbage()
          end 

          self.naNet:zeroGradParameters()
          self:docGrad(d,naTrData,trTargets)
          cmdAdaGradC(self.naNet:get(1).weight, self.naNet:get(1).gradWeight,
                      eta, lamb, self.nastates[1])
          cmdAdaGradC(self.naNet:get(3).bias, self.naNet:get(3).gradBias, 
                      eta, lamb, self.nastates[2])  
          cmdAdaGradC(self.naNet:get(5).weight, self.naNet:get(5).gradWeight, 
                      eta, lamb, self.nastates[3])                         
          cmdAdaGradC(self.naNet:get(5).bias, self.naNet:get(5).gradBias, 
                      eta, lamb, self.nastates[4])
                 
       end
  
       print("evaluating on dev...")
       local prec,rec,currF = self:getDevF(naDevData,devTargets)
       print(string.format("P/R/F: %f/%f/%f", prec,rec,currF))
       if currF > bestF then
         bestF = currF
         if save then
           print("overwriting params...")
           torch.save(serFi .. string.format("-na-%f-%f",eta,lamb), self.naNet)
         end
       end
       collectgarbage()
    end 
end


function ANAModel:getDevF(naDevData,devTargets)
    local tp = 0
    local tn = 0
    local fp = 0
    local fn = 0
    for d = 1, naDevData.numDocs do
        if d % 100 == 0 then
            print("dev doc " .. tostring(d))
            collectgarbage()
        end
        local numMents = naDevData:numMents(d)
        for m = 2, numMents do
           local mentFeats = naDevData:getFeats(d,m)
           -- need to make 1 x numFeats for our network to work...
           local score = self.naNet:forward(mentFeats:view(1,mentFeats:size(1)))[1][1]
           if devTargets[d][m-1] == 1 then
               if score >= 0 then
                   tp = tp + 1
               else
                   fn = fn + 1
               end
           else
               if score < 0 then
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

function makeTargets(naSpData,clusts)
    local targets = {}
    for i = 1, naSpData.numDocs do
        t = torch.ones(naSpData:numMents(i)-1)
        for j = 2, naSpData:numMents(i) do
            if not clusts[i]:anaphoric(j) then
                t[j-1] = -1
            end
        end
        table.insert(targets,t)
    end
    return targets   
end

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training anaph model')
cmd:text()
cmd:text('Options')
cmd:option('-hidden_unary', 128, 'Hidden layer size')
cmd:option('-trainClustFi', '../TrainOPCs.txt', 'Train Oracle Predicted Clustering File')
cmd:option('-devClustFi', '../DevOPCs.txt', 'Dev Oracle Predicted Clustering File')
cmd:option('-anaTrFeatPfx', 'train_basicp', 'expects train features in <anaTrFeatPfx>-na-*.h5')
cmd:option('-anaDevFeatPfx', 'dev_basicp', 'expects dev features in <anaDevFeatPfx>-na-*.h5')
cmd:option('-c0', 1, 'false positive cost')
cmd:option('-c1', 1.4, 'false negative cost')
cmd:option('-nEpochs', 10, 'number of epochs to train')
cmd:option('-save', false, 'save best model')
cmd:option('-savePfx', '', 'prefixes saved model with this')
cmd:option('-t', 2, "Number of threads")
cmd:option('-eta', 0.1, 'adagrad learning rate')
cmd:option('-lamb', 0.00001, 'l1 regularization coefficient')
cmd:text()

-- Parse input options
options = cmd:parse(arg)

if options.t > 0 then
    torch.setnumthreads(options.t)
end
print("Using " .. tostring(torch.getnumthreads()) .. " threads")


function main()
    local anaTrData = SpDMData.loadFromH5(options.anaTrFeatPfx)
    print("read anaph train data")
    print("max ana feature is: " .. anaTrData.maxFeat)
    local anaDevData = SpDMData.loadFromH5(options.anaDevFeatPfx)
    print("read anaph dev data") 
    local trClusts = getOPCs(options.trainClustFi,anaTrData)
    print("read train clusters")
    local devClusts = getOPCs(options.devClustFi,anaDevData)
    print("read dev clusters")
    -- make targets for training     
    local trTargets = makeTargets(anaTrData,trClusts)
    local devTargets = makeTargets(anaDevData,devClusts)

    local anaModel = ANAModel.make(anaTrData.maxFeat, options.hidden_unary, options.c0, options.c1)
    anaModel:train(anaTrData,trTargets,anaDevData,devTargets,options.eta,options.lamb,
                   options.nEpochs, options.save, options.savePfx)
end

main()
