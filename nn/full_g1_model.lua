require 'nn'
dofile('init.lua')
require 'cmd_adagrad'
require 'coref_utils'
require 'sparse_doc_data'

FullG1Model = {}
FullG1Model.__index = FullG1Model

torch.manualSeed(2)

function FullG1Model.make(pwD, hiddenPW, naD, hiddenUnary, fl, fn, wl) 
    torch.manualSeed(2)
    model = {}
    setmetatable(model,FullG1Model)
    model.hiddenPW = hiddenPW
    model.hiddenUnary = hiddenUnary
    model.fl = fl
    model.fn = fn
    model.wl = wl

    -- make net for scoring non-anaphoric case
    local naNet = nn.Sequential()
    naNet:add(nn.LookupTable(naD,hiddenUnary))   
    naNet:add(nn.Sum(1))
    naNet:add(nn.Add(hiddenUnary)) -- add a bias
    naNet:add(nn.Tanh())
    naNet:add(nn.Linear(hiddenUnary,1))
    model.naNet = naNet 
    -- make sure contiguous, and do sparse sutskever init while we're at it
    checkContigAndSutsInit(naNet,15)    
    model.nastates = {{},{},{},{}} 
    collectgarbage()

    -- make net for scoring anaphoric case
    local pwNet = nn.Sequential()
    local firstLayer = nn.ParallelTable() -- joins anaphoricity and pairwise representations
    local anteNet = nn.Sequential()
    anteNet:add(nn.LookupTable(pwD,hiddenPW))
    anteNet:add(nn.Sum(1)) -- equivalent to a sparse Mat vec product
    anteNet:add(nn.Add(hiddenPW)) -- add a bias
    checkContigAndSutsInit(anteNet,15)
    collectgarbage()
    model.anteNet = anteNet -- just adding for convenience
    firstLayer:add(anteNet)
    
    -- anaNet has the same architecture as naNet
    local anaNet = nn.Sequential()
    anaNet:add(nn.LookupTable(naD,hiddenUnary))  
    anaNet:add(nn.Sum(1))
    anaNet:add(nn.Add(hiddenUnary)) -- add a bias

    -- naNet and anaNet should share first layer weight and bias...
    anaNet:share(naNet,'weight','bias','gradWeight','gradBias')
    firstLayer:add(anaNet)
    model.anaNet = anaNet
    pwNet:add(firstLayer)
    pwNet:add(nn.JoinTable(1,1))

    pwNet:add(nn.Tanh())
    pwNet:add(nn.Linear(hiddenPW+hiddenUnary,1))
    model.pwNet = pwNet  
    -- initialize second layer bias
    pwNet:get(4).bias:fill(0.5)
    
    model.pwstates = {{},{},{},{}}
    collectgarbage()
    return model
end 


-- initializes with pre-trained parameters
function FullG1Model.preInit(fl, fn, wl, anteSerFi, anaSerFi) 
    torch.manualSeed(2)
    model = {}
    setmetatable(model,FullG1Model)
    model.fl = fl
    model.fn = fn
    model.wl = wl

    -- make net for scoring non-anaphoric case
    local naNet = torch.load(anaSerFi)
    model.naNet = naNet 
    -- we keep the first layer, but reinitialize the last layer (v in our notation)
    naNet.modules[2] = nn.Sum(1) -- used to sum over 2, b/c were training in batches, but now just 1
    collectgarbage()
    -- re-initialize v (apart from bias) in the same way torch does...
    local stdv = 1./math.sqrt(naNet:get(5).weight:size(2))
    naNet:get(5).weight:uniform(-stdv,stdv) 
    naNet:get(5).bias:fill(0.5)
    checkContig(naNet)

    model.nastates = {{},{},{},{}}
    local naD = naNet:get(1).weight:size(1)
    model.hiddenUnary = naNet:get(1).weight:size(2)
    collectgarbage()

    -- make net for scoring anaphoric case
    local pwNet = nn.Sequential()
    local firstLayer = nn.ParallelTable() -- joins anaphoricity and pairwise features
    local anteNet = torch.load(anteSerFi)
    -- we only want the first 3 layers from pre-training
    anteNet.modules[4] = nil -- would've been Tanh
    anteNet.modules[5] = nil -- would've been final linear layer
    collectgarbage()
    checkContig(anteNet)
    model.hiddenPW = anteNet:get(1).weight:size(2)
    
    model.anteNet = anteNet -- just adding for convenience
    firstLayer:add(anteNet)
    -- anaNet has the same architecture as naNet
    local anaNet = nn.Sequential()
    anaNet:add(nn.LookupTable(naD,model.hiddenUnary))  
    anaNet:add(nn.Sum(1))
    anaNet:add(nn.Add(model.hiddenUnary)) -- add a bias
    
    -- naNet and anaNet should share first layer weight and bias...
    anaNet:share(naNet,'weight','bias','gradWeight','gradBias')
    firstLayer:add(anaNet)
    model.anaNet = anaNet
    pwNet:add(firstLayer)
    pwNet:add(nn.JoinTable(1,1))
    pwNet:add(nn.Tanh())
    pwNet:add(nn.Linear(model.hiddenPW+model.hiddenUnary,1))
    model.pwNet = pwNet  
    
    -- initialize second layer bias
    pwNet:get(4).bias:fill(0.5)
    assert(pwNet:get(4).weight:isContiguous())
    assert(pwNet:get(4).bias:isContiguous())
    
    model.pwstates = {{},{},{},{}}
    collectgarbage()
    return model
end 


function FullG1Model.load(pwFullSerFi, naFullSerFi, fl, fn, wl) 
    torch.manualSeed(2)
    model = {}
    setmetatable(model,FullG1Model)
    model.fl = fl
    model.fn = fn
    model.wl = wl

    local naNet = torch.load(naFullSerFi)
    model.naNet = naNet 
    model.nastates = {{},{},{},{}}
    local naD = naNet:get(1).weight:size(1)
    model.hiddenUnary = naNet:get(1).weight:size(2)
    collectgarbage()

    local pwNet = torch.load(pwFullSerFi)
    model.pwNet = pwNet
    model.anteNet = pwNet:get(1):get(1)
    model.hiddenPW = model.anteNet:get(1).weight:size(2)
    model.anaNet = pwNet:get(1):get(2)
    
    -- naNet and anaNet should share first layer weight and bias...
    model.anaNet:share(model.naNet,'weight','bias','gradWeight','gradBias')
    
    model.pwstates = {{},{},{},{}}
    collectgarbage()
    return model
end 


function FullG1Model:docGrad(d,pwData,anaData,clusts,pwnz,nanz)
    local numMents = pwData:numMents(d)
    local numPairs = (numMents*(numMents+1))/2
    local mistakes = {} -- keeps track of stuff we'll use to update gradient
    local deltTensor = torch.zeros(1) -- will hold delta for backprop
    -- calculate all pairwise scores in batch
    local scores = self:docBatchFwd(d,numMents,numPairs,pwData,anaData)
    for m = 2, numMents do -- ignore first guy; always NA       
        local start = ((m-1)*m)/2 -- one behind first pair for mention m
        -- score NA case separately
        scores[start+m] = self.naNet:forward(anaData:getFeats(d,m))
        -- pick a latent antecedent
        local late = m
        if clusts[d]:anaphoric(m) then
           late = maxGoldAnt(clusts[d],scores,m,start)
        end
                     
        local pred = scores.cr.CR_mult_la_argmax(m,late,start,scores,clusts[d].clusts[clusts[d].m2c[m]],clusts[d].m2c,self.fl,self.fn,self.wl)
                
        local delt = clusts[d]:cost(m,pred,self.fl,self.fn,self.wl)
            
        if delt > 0 then
            deltTensor[1] = delt
            -- gradients involve adding predicted thing and subtracting latent thing
            if pred ~= m then
                -- run the predicted thing thru the net again so we can backprop
                self.pwNet:forward({pwData:getFeats(d,m,pred),anaData:getFeats(d,m)})
                self.pwNet:backward({pwData:getFeats(d,m,pred),anaData:getFeats(d,m)},deltTensor)
            else
                -- the predicted thing must have been m, which we already ran thru naNet
                self.naNet:backward(anaData:getFeats(d,m),deltTensor) 
            end
            -- same deal for subtracting latent thing
            if late ~= m then
                self.pwNet:forward({pwData:getFeats(d,m,late),anaData:getFeats(d,m)})
                self.pwNet:backward({pwData:getFeats(d,m,late),anaData:getFeats(d,m)},-deltTensor)            
            else
                self.naNet:backward(anaData:getFeats(d,m),-deltTensor)
            end
            table.insert(mistakes,{m,pred,late})
        end  
    end
    -- update nz idxs
    for i, mistake in ipairs(mistakes) do
        local ment = mistake[1]
        addKeys(anaData:getFeats(d,ment),nanz)
        if mistake[2] ~= ment then
            addKeys(pwData:getFeats(d,ment,mistake[2]),pwnz)
        end
        if mistake[3] ~= ment then
            addKeys(pwData:getFeats(d,ment,mistake[3]),pwnz)
        end
    end
end


function FullG1Model:docBatchFwd(d,numMents,numPairs,pwData,anaData)
    local onebuf = torch.ones(numPairs)
    local Z1 = torch.zeros(self.hiddenPW+self.hiddenUnary,numPairs) -- score everything at once
    -- start by adding biases
    Z1:sub(1,self.hiddenPW):addr(0,1,self.anteNet:get(3).bias,onebuf)
    Z1:sub(self.hiddenPW+1,self.hiddenPW+self.hiddenUnary):addr(0,1,self.naNet:get(3).bias,onebuf)
    -- now do sparse mult and tanh over all pairs: gives something hidden_pairwise x numPairs
    Z1.cr.CR_fm_layer1(Z1,self.anteNet:get(1).weight,pwData.feats,pwData.mentStarts,self.naNet:get(1).weight,anaData.feats,anaData.mentStarts,pwData.docStarts[d],anaData.docStarts[d],numMents)
    local scores = torch.mv(Z1:t(),self.pwNet:get(4).weight[1])
    scores:add(self.pwNet:get(4).bias[1])
    return scores
end


-- trains from random init
function train(pwData,anaData,clusts,pwDevData,anaDevData,hiddenPW,hiddenUnary,fl,fn,wl,
                  eta1,eta2,lamb,nEpochs,save,savePfx,PT,anteSerFi,anaSerFi)
   local PT = PT or true -- whether to initialize with pretrained params
   local nEpochs = nEpochs or 5
   local serFi = string.format("models/%s-%f-%f-%f.model", savePfx, fl, fn, wl)

   torch.manualSeed(2)
   local fm = nil
   if PT then
     fm = FullG1Model.preInit(fl, fn, wl, anteSerFi, anaSerFi)
   else
     fm = FullG1Model.make(pwData.maxFeat, hiddenPW, anaData.maxFeat, hiddenUnary, fl, fn, wl)
   end
   collectgarbage()

   for n = 1, nEpochs do
      print("epoch: " .. tostring(n))
      -- use document sized minibatches
      for d = 1, pwData.numDocs do
         if d % 200 == 0 then
            print("doc " .. tostring(d))
            collectgarbage()
         end 
         local pwnz = {}
         local nanz = {}     
 
         fm.pwNet:zeroGradParameters()
         fm.naNet:zeroGradParameters()
         fm:docGrad(d,pwData,anaData,clusts,pwnz,nanz)
         
         -- update pw parameters
         colTblCmdAdaGrad(fm.anteNet:get(1).weight, fm.anteNet:get(1).gradWeight, pwnz,
                     eta1, lamb, fm.pwstates[1])
         cmdAdaGradC(fm.anteNet:get(3).bias, fm.anteNet:get(3).gradBias, 
                     eta1, lamb, fm.pwstates[2])  
         cmdAdaGradC(fm.pwNet:get(4).weight, fm.pwNet:get(4).gradWeight, 
                     eta2, lamb, fm.pwstates[3])                         
         cmdAdaGradC(fm.pwNet:get(4).bias, fm.pwNet:get(4).gradBias, 
                     eta2, lamb, fm.pwstates[4])
          
         -- update ana parameters       
         colTblCmdAdaGrad(fm.naNet:get(1).weight, fm.naNet:get(1).gradWeight, nanz,
                     eta1, lamb, fm.nastates[1])
         cmdAdaGradC(fm.naNet:get(3).bias, fm.naNet:get(3).gradBias, 
                     eta1, lamb, fm.nastates[2])
         cmdAdaGradC(fm.naNet:get(5).weight, fm.naNet:get(5).gradWeight, 
                     eta2, lamb, fm.nastates[3])
         cmdAdaGradC(fm.naNet:get(5).bias, fm.naNet:get(5).gradBias, 
                     eta2, lamb, fm.nastates[4])                       
      end

      if save then
         print("overwriting params...")
         if PT then
           torch.save(serFi.."-full_g1_PT-pw",fm.pwNet)
           torch.save(serFi.."-full_g1_PT-na",fm.naNet)
         else
           torch.save(serFi.."-full_g1_RI-pw",fm.pwNet)
           torch.save(serFi.."-full_g1_RI-na",fm.naNet)
         end
      end

      collectgarbage()
   end 
end

-- writes backptrs so we can turn them into CoNLL fmt predictions and call the script
function FullG1Model:writeBPs(pwDevData,anaDevData,bpfi)
    local bpfi = bpfi or "bps/dev.bps"
    local ofi = assert(io.open(bpfi,"w"))
    for d = 1, pwDevData.numDocs do
        if d % 100 == 0 then
            print("dev doc " .. tostring(d))
            collectgarbage()
        end
        local numMents = anaDevData:numMents(d)
        local numPairs = (numMents*(numMents+1))/2
        ofi:write("0") -- predict first thing links to itself always
        local scores = self:docBatchFwd(d,numMents,numPairs,pwDevData,anaDevData)
        for m = 2, numMents do
            local start = ((m-1)*m)/2
            -- rescore NA case (batch only does anaphoric cases)
            scores[start+m] = self.naNet:forward(anaDevData:getFeats(d,m))
            local _, pred = torch.max(scores:sub(start+1,start+m),1)
            ofi:write(" ",tostring(pred[1]-1))
        end 
        ofi:write("\n")
    end
    ofi:close()
end


function FullG1Model:docLoss(d,pwData,anaData,clusts)
    local loss = 0
    local numMents = pwData:numMents(d)
    local numPairs = (numMents*(numMents+1))/2
    local scores = self:docBatchFwd(d,numMents,numPairs,pwData,anaData)
    for m = 2, numMents do -- ignore first guy; always NA
        local start = (m*(m-1))/2
        scores[start+m] = self.naNet:forward(anaData:getFeats(d,m))
        local late = m
        if clusts[d]:anaphoric(m) then
           late = maxGoldAnt(clusts[d],scores,m,start)
        end

        local pred = scores.cr.CR_mult_la_argmax(m,late,start,scores,clusts[d].clusts[clusts[d].m2c[m]],clusts[d].m2c,self.fl,self.fn,self.wl)        
        local delt = clusts[d]:cost(m,pred,self.fl,self.fn,self.wl)
        if delt > 0 then
            loss = loss + delt*(1 + scores[start+pred] - scores[start+late])
        end
    end
    return loss
end


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training full g1 model')
cmd:text()
cmd:text('Options')
cmd:option('-hidden_unary', 128, 'Hidden layer.')
cmd:option('-hidden_pairwise', 700, 'Hidden layer.')
cmd:option('-trainClustFi', '../TrainOPCs.txt', 'Train Oracle Predicted Clustering File')
cmd:option('-pwTrFeatPfx', 'train_basicp', 'expects train features in <pwTrFeatPfx>-pw-*.h5')
cmd:option('-pwDevFeatPfx', 'dev_basicp', 'expects dev features in <pwDevFeatPfx>-pw-*.h5')
cmd:option('-anaTrFeatPfx', 'train_basicp', 'expects train features in <anaTrFeatPfx>-na-*.h5')
cmd:option('-anaDevFeatPfx', 'dev_basicp', 'expects dev features in <anaDevFeatPfx>-na-*.h5')
cmd:option('-antePTSerFi','models/basicp_700.model-pw-0.100000-0.000010','pretrained antecedent network')
cmd:option('-anaphPTSerFi','models/basicp_128.model-na-0.100000-0.000010','pretrained anaphoricity network')
cmd:option('-random_init', false, 'randomly initialize parameters')
cmd:option('-nEpochs', 14, 'number of epochs to train')
cmd:option('-fl', 0.5, 'False Link cost')
cmd:option('-fn', 1.2, 'False New cost')
cmd:option('-wl', 1, 'Wrong Link cost')
cmd:option('-t', 8, "Number of threads")
cmd:option('-eta1', 0.1, 'adagrad learning rate for first layer')
cmd:option('-eta2', 0.001, 'adagrad learning rate for second layer')
cmd:option('-lamb', 0.000001, 'l1 regularization coefficient')
cmd:option('-save', false, 'save model')
cmd:option('-savePfx', '', 'prefixes saved model with this')
cmd:option('-load_and_predict', false, 'load full model and predict (on dev or test)')
cmd:option('-pwFullSerFi', 'models/basicp-0.500000-1.200000-1.000000.model-full_g1_PT-pw', 'serialized pairwise network')
cmd:option('-anaFullSerFi', 'models/basicp-0.500000-1.200000-1.000000.model-full_g1_PT-na', 'serialized anaphoricity network')
cmd:text()

-- Parse input options
opts = cmd:parse(arg)

if opts.t > 0 then
    torch.setnumthreads(opts.t)
end
print("Using " .. tostring(torch.getnumthreads()) .. " threads")


function main()
    if not opts.load_and_predict then -- if training, get train data
       local pwTrData = SpDMPWData.loadFromH5(opts.pwTrFeatPfx)
       print("read pw train data")
       print("max pw feature is: " .. pwTrData.maxFeat)
       local anaTrData = SpDMData.loadFromH5(opts.anaTrFeatPfx)
       print("read anaph train data")
       print("max ana feature is: " .. anaTrData.maxFeat)       
       local trClusts = getOPCs(opts.trainClustFi,anaTrData)
       print("read train clusters")   
       local pwDevData = SpDMPWData.loadFromH5(opts.pwDevFeatPfx)
       print("read pw dev data")
       local anaDevData = SpDMData.loadFromH5(opts.anaDevFeatPfx)
       print("read anaph dev data")  
       train(pwTrData,anaTrData,trClusts,pwDevData,anaDevData,opts.hidden_pairwise,
            opts.hidden_unary,opts.fl,opts.fn,opts.wl,opts.eta1,opts.eta2,
            opts.lamb,opts.nEpochs,opts.save,opts.savePfx,
            (not opts.random_init),opts.antePTSerFi,opts.anaphPTSerFi)    
    else
       local pwDevData = SpDMPWData.loadFromH5(opts.pwDevFeatPfx)
       print("read pw dev data")
       local anaDevData = SpDMData.loadFromH5(opts.anaDevFeatPfx)
       print("read anaph dev data")
      local fm = FullG1Model.load(opts.pwFullSerFi, opts.anaFullSerFi)
      fm:writeBPs(pwDevData,anaDevData,"bps/load_and_pred.bps")
    end
end

main()

