require 'nn'
dofile('init.lua')
require 'cmd_adagrad'
require 'coref_utils'
require 'sparse_doc_data'

AnteModel = {}
AnteModel.__index = AnteModel

torch.manualSeed(2)

function AnteModel.make(pwD, hiddenPW) 
    torch.manualSeed(2)
    model = {}
    setmetatable(model,AnteModel)
    model.hiddenPW = hiddenPW
    model.fl = 1
    model.fn = 1
    model.wl = 1
    
    local pwNet = nn.Sequential()
    pwNet:add(nn.LookupTable(pwD,hiddenPW))
    pwNet:add(nn.Sum(1)) -- equivalent to a sparse Mat vec product
    pwNet:add(nn.Add(hiddenPW)) -- add a bias
    pwNet:add(nn.Tanh())
    pwNet:add(nn.Linear(hiddenPW,1))    
    model.pwNet = pwNet
    
    -- make sure contiguous, and do sparse Sutskever init while we're at it
    checkContigAndSutsInit(pwNet,15)
    model.pwstates = {{},{},{},{}}
    
    collectgarbage()
    return model
end 


function AnteModel.load(hiddenPW, serFi) 
    torch.manualSeed(2)
    model = {}
    setmetatable(model,AnteModel)
    model.hiddenPW = hiddenPW
    model.fl = 1
    model.fn = 1
    model.wl = 1 
    model.pwNet = torch.load(serFi)
    model.pwstates = {{},{},{},{}}
    
    collectgarbage()
    return model
end 

--[[ 
   - computes gradients assuming parameters have been zeroed out
   - pwData is a SpDMPWData
--]]    
function AnteModel:docGrad(d,pwData,clusts,pwnz)
    local numMents = pwData:numMents(d)
    local numPairs = (numMents*(numMents+1))/2
    local mistakes = {} -- keeps track of stuff we'll use to update gradient
    local deltTensor = torch.ones(1) -- will hold delta for backprop
    -- calculate all pairwise scores (b/c so few are anaphoric, quicker to not batch...)
    --local scores = self:docBatchFwd(d,numMents,numPairs,pwData)
    local scores = torch.zeros(numPairs)

    for m = 2, numMents do -- ignore first guy; always NA
        if clusts[d]:anaphoric(m) then        
            local start = ((m-1)*m)/2 -- one behind first pair for mention m
         
         -- score all the potential antecedents
            for a = 1, m-1 do
                scores[start+a] = self.pwNet:forward(pwData:getFeats(d,m,a))
            end
            
            local late = maxGoldAnt(clusts[d],scores,m,start)
            scores[start+m] = -math.huge -- this is a hack; the method below looks thru m
            local pred = scores.cr.CR_mult_la_argmax(m,late,start,scores,clusts[d].clusts[clusts[d].m2c[m]],clusts[d].m2c,self.fl,self.fn,self.wl)

            local delt = clusts[d]:cost(m,pred,self.fl,self.fn,self.wl) -- will always be 1 or 0
            
            if delt > 0 then
                -- gradients wrt params will essentially involve adding pred and subtracting latent
                -- run the predicted thing thru the net again so we can backprop
                self.pwNet:forward(pwData:getFeats(d,m,pred))
                self.pwNet:backward(pwData:getFeats(d,m,pred),deltTensor)
                self.pwNet:forward(pwData:getFeats(d,m,late))
                self.pwNet:backward(pwData:getFeats(d,m,late),-deltTensor)
                table.insert(mistakes,{m,pred,late})
            end
        end
    end
    -- update nz idxs
    for i, mistake in ipairs(mistakes) do
        addKeys(pwData:getFeats(d,mistake[1],mistake[2]),pwnz)
        addKeys(pwData:getFeats(d,mistake[1],mistake[3]),pwnz)
    end
end


function AnteModel:docBatchFwd(d,numMents,numPairs,pwData)
    local onebuf = torch.ones(numPairs)
    local Z1 = torch.zeros(self.hiddenPW,numPairs) -- score everything at once
    -- start by adding bias
    Z1:addr(0,1,self.pwNet:get(3).bias,onebuf)
    -- now do sparse mult over all pairs: gives something hidden_pairwise x numPairs
    Z1.cr.CR_sparse_lt_mult(Z1,self.pwNet:get(1).weight,pwData.feats,pwData.mentStarts,pwData.docStarts[d],numMents)
    Z1:tanh()
    scores = torch.mv(Z1:t(),self.pwNet:get(5).weight[1])
    scores:add(self.pwNet:get(5).bias[1])
    return scores
end


function AnteModel:train(pwData,clusts,pwDevData,devClusts,eta,lamb,nEpochs,
                        save,savePfx)
    local nEpochs = nEpochs or 5
    local serFi = string.format("models/%s_%d.model", savePfx, self.hiddenPW)
             
    -- reset stuff
    torch.manualSeed(2)
    self.pwstates = {{},{},{},{}}
    collectgarbage()           
    checkContigAndSutsInit(self.pwNet,15)
    collectgarbage()
       
    local bestAcc = 0
    for n = 1, nEpochs do
       print("epoch: " .. tostring(n))
       -- use document sized minibatches
       for d = 1, pwData.numDocs do
          if d % 200 == 0 then
             print("doc " .. tostring(d))
             collectgarbage()
          end 
          local pwnz = {}
          self.pwNet:zeroGradParameters()
          self:docGrad(d,pwData,clusts,pwnz)
          -- do pw gradients
          colTblCmdAdaGrad(self.pwNet:get(1).weight, self.pwNet:get(1).gradWeight, pwnz,
                      eta, lamb, self.pwstates[1])
          cmdAdaGradC(self.pwNet:get(3).bias, self.pwNet:get(3).gradBias, 
                      eta, lamb, self.pwstates[2])  
          cmdAdaGradC(self.pwNet:get(5).weight, self.pwNet:get(5).gradWeight, 
                      eta, lamb, self.pwstates[3])                         
          cmdAdaGradC(self.pwNet:get(5).bias, self.pwNet:get(5).gradBias, 
                      eta, lamb, self.pwstates[4])   
       end
  
       print("evaluating on dev...")
       local currAcc = self:getDevAcc(pwDevData,devClusts)
       print("Acc " .. tostring(currAcc))
       print("")
       if currAcc > bestAcc then
         bestAcc = currAcc
         if save then
           print("overwriting params...")
           torch.save(serFi..string.format("-pw-%f-%f",eta, lamb) ,self.pwNet)
         end
       end
       collectgarbage()
    end
end


function AnteModel:getDevAcc(pwDevData,devClusts)
    local total = 0
    local correct = 0
    for d = 1, pwDevData.numDocs do
        if d % 100 == 0 then
            print("dev doc " .. tostring(d))
            collectgarbage()
        end
        local numMents = pwDevData:numMents(d)
        local numPairs = (numMents*(numMents+1))/2
        --local scores = self:docBatchFwd(d,numMents,numPairs,pwDevData)
        local scores = torch.zeros(numPairs)
        for m = 2, numMents do -- automatically assign m = 1 to itself (w/ ones above)
            if devClusts[d]:anaphoric(m) then
                local start = ((m-1)*m)/2 
                for a = 1, m-1 do
                  scores[start+a] = self.pwNet:forward(pwDevData:getFeats(d,m,a))
                end
                local _, pred = torch.max(scores:sub(start+1,start+m-1),1)
                total = total + 1
                if devClusts[d].m2c[m] == devClusts[d].m2c[pred[1]] then
                    correct = correct + 1
                end
            end 
        end
    end
    return correct/total
end


function AnteModel:docLoss(d,pwData,clusts)
    local loss = 0
    local numMents = pwData:numMents(d)
    local numPairs = (numMents*(numMents+1))/2
    local scores = torch.zeros(numPairs)
    for m = 2, numMents do -- ignore first guy; always NA
        if clusts[d]:anaphoric(m) then
            local start = (m*(m-1))/2 -- index one behind first antecedent for this mention (in pwData)
            -- score all the potential antecedents
            for a = 1, m-1 do
                scores[start+a] = self.pwNet:forward(pwData:getFeats(d,m,a))
            end
            local late = maxGoldAnt(clusts[d],scores,m,start)
            scores[start+m] = -math.huge -- this is a hack; the method below looks thru m
            local pred = scores.cr.CR_mult_la_argmax(m,late,start,scores,clusts[d].clusts[clusts[d].m2c[m]],clusts[d].m2c,self.fl,self.fn,self.wl)

            local delt = clusts[d]:cost(m,pred,self.fl,self.fn,self.wl)
            if delt > 0 then
                loss = loss + delt*(1 + scores[start+pred] - scores[start+late])
            end
        end
    end
    return loss
end

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training ante model')
cmd:text()
cmd:text('Options')
cmd:option('-hidden_pairwise', 700, 'Hidden layer size')
cmd:option('-trainClustFi', '../TrainOPCs.txt', 'Train Oracle Predicted Clustering File')
cmd:option('-devClustFi', '../DevOPCs2012.txt', 'Dev Oracle Predicted Clustering File')
cmd:option('-pwTrFeatPfx', 'train_basicp', 'expects train features in <pwTrFeatPfx>-pw-*.h5')
cmd:option('-pwDevFeatPfx', 'dev_basicp', 'expects dev features in <pwDevFeatPfx>-pw-*.h5')
cmd:option('-nEpochs', 4, 'number of epochs to train')
cmd:option('-save', false, 'save best model')
cmd:option('-savePfx', '', 'prefixes saved model with this')
cmd:option('-t', 8, "Number of threads")
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
    local pwTrData = SpDMPWData.loadFromH5(options.pwTrFeatPfx)
    print("read pw train data")
    print("max pw feature is: " .. pwTrData.maxFeat)
    local pwDevData = SpDMPWData.loadFromH5(options.pwDevFeatPfx)
    print("read pw dev data")
    local trClusts = getOPCs(options.trainClustFi,pwTrData)
    print("read train clusters")
    local devClusts = getOPCs(options.devClustFi,pwDevData)
    print("read dev clusters")    
   
    local anteModel = AnteModel.make(pwTrData.maxFeat, options.hidden_pairwise)
    anteModel:train(pwTrData,trClusts,pwDevData,devClusts,options.eta,options.lamb,
       options.nEpochs,options.save,options.savePfx)
   
end

main()
