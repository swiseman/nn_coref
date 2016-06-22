require 'nn'
require 'coref_utils'
require 'sparse_doc_data'

local mu = require 'model_utils'

do

  local VanillaMR = torch.class('VanillaMR')

  function VanillaMR:__init(pwD, Hp, anaD, Ha, fl, fn, wl, cuda, anteSerFi, anaSerFi, dop) 
    torch.manualSeed(2)
    if cuda then
      cutorch.manualSeed(2)
    end

    self.fl = fl
    self.fn = fn
    self.wl = wl

    local naNet = mu.make_sp_mlp(anaD,Ha,true,false)
    if anaSerFi then
      print("preinit'ing naNet with " .. anaSerFi)
      local preNaNet = torch.load(anaSerFi)
      -- for some reason copying makes stuff faster
      assert(naNet:get(1).weight:size(1) == preNaNet:get(1).weight:size(1)
           and naNet:get(1).weight:size(2) == preNaNet:get(1).weight:size(2))
      for i = 1, naNet:get(1).weight:size(1) do
        naNet:get(1).weight[i] = preNaNet:get(1).weight[i]
      end
      naNet:get(3).bias = preNaNet:get(3).bias:clone()
    end
        -- remove dropout
    naNet.modules[5] = naNet.modules[6]
    naNet.modules[6] = nil
    collectgarbage()

    -- make net for scoring anaphoric case
    local pwNet = nn.Sequential()
    local firstLayer = nn.ParallelTable() -- joins all reps
    local anteNet = mu.make_sp_mlp(pwD,Hp,true,true)
    if anteSerFi then
      print("preinit'ing anteNet with " .. anteSerFi)
      local preAnteNet = torch.load(anteSerFi)
      assert(anteNet:get(1).weight:size(1) == preAnteNet:get(1).weight:size(1)
           and anteNet:get(1).weight:size(2) == preAnteNet:get(1).weight:size(2))
      for i = 1, anteNet:get(1).weight:size(1) do
        anteNet:get(1).weight[i] = preAnteNet:get(1).weight[i]:float()
      end
      anteNet:get(3).bias = preAnteNet:get(3).bias:float():clone()    
    end

    firstLayer:add(anteNet)
    -- add clone of naNet
    local naNetClone = naNet:clone('weight','bias','gradWeight','gradBias')
    -- get rid of dropout and final linear
    naNetClone.modules[5] = nil
    naNetClone.modules[6] = nil
    collectgarbage()
    firstLayer:add(naNetClone)
    
    pwNet:add(firstLayer)
    pwNet:add(nn.JoinTable(2,2))
    pwNet:add(nn.Dropout(dop))
    pwNet:add(nn.Linear(Hp+Ha,1))
        
    naNet:get(1).weight[-1]:fill(0)
    pwNet:get(1):get(1):get(1).weight[-1]:fill(0)
    
    self.naNet = cuda and naNet:cuda() or naNet
    self.pwNet = cuda and pwNet:cuda() or pwNet
    
    -- need to reshare anaphoricity feature weights (if cuda'ing)
    self.pwNet:get(1):get(2):get(1):share(self.naNet:get(1), 'weight', 'gradWeight')
    self.pwNet:get(1):get(2):get(3):share(self.naNet:get(3), 'bias', 'gradBias')
    collectgarbage()
  end 
    
  
  function VanillaMR:docGrad(d,pwDocBatch,anaDocBatch,OPC,deltTensor,numMents,cdb)

    for m = 2, numMents do -- ignore first guy; always NA 
      local cid = OPC.m2c[m]
      local start = ((m-2)*(m-1))/2 -- one behind first pair for mention m
      
      local scores = self.pwNet:forward({pwDocBatch:sub(start+1,start+m-1),
          anaDocBatch:sub(m,m):expand(m-1,anaDocBatch:size(2))}):squeeze(2)
      local naScore = self.naNet:forward(anaDocBatch:sub(m,m)):squeeze() -- always 1x1
      -- pick a latent antecedent
      local late = m
      local lateScore = naScore
      if OPC:anaphoric(m) then
        late = maxGoldAnt(OPC,scores,m,0)
        lateScore = scores[late]
      end

      local pred, delt = simpleMultLAArgmax(OPC,scores,m,lateScore,naScore,0,self.fl,self.fn,self.wl)
          
      if delt > 0 then
        deltTensor[1][1] = delt
        -- gradients involve adding predicted thing and subtracting latent thing
        if pred ~= m then
          -- run the predicted thing thru the net again so we can backprop
          self.pwNet:forward({pwDocBatch:sub(start+pred,start+pred),
                    anaDocBatch:sub(m,m)})
          self.pwNet:backward({pwDocBatch:sub(start+pred,start+pred),
                    anaDocBatch:sub(m,m)}, deltTensor)          
        else
          -- the predicted thing must have been m, which we already ran thru naNet
          self.naNet:backward(anaDocBatch:sub(m,m),deltTensor) 
        end
        -- same deal for subtracting latent thing
        if late ~= m then          
          self.pwNet:forward({pwDocBatch:sub(start+late,start+late),
                    anaDocBatch:sub(m,m)})
          self.pwNet:backward({pwDocBatch:sub(start+late,start+late),
                    anaDocBatch:sub(m,m)}, -deltTensor) 
        else
          self.naNet:backward(anaDocBatch:sub(m,m),-deltTensor)
        end
      end  -- end if delt > 0
    end -- end for m
  end

end

do
  local SavedVanillaMR, parent = torch.class('SavedVanillaMR')

  function SavedVanillaMR:__init(naNetFi, pwNetFi, cuda) 
    torch.manualSeed(2)
    if cuda then
      cutorch.manualSeed(2)
    end
     
    -- forget cuda for now...
    self.cuda = cuda
    self.naNet = cuda and torch.load(naNetFi) or torch.load(naNetFi):float()
    self.pwNet = cuda and torch.load(pwNetFi) or torch.load(pwNetFi):float()
    collectgarbage()
  end 
  
  
  -- writes backptrs so we can turn them into CoNLL fmt predictions and call the script
  -- note this is misnamed...whatever
  function SavedVanillaMR:writeBPs(pwDevData,anaDevData,cuda,bpfi)
    local bpfi = bpfi or "bps/dev.bps"
    local ofi = assert(io.open(bpfi,"w"))
    assert(self.naNet:get(1).weight[-1]:abs():sum() == 0)
    assert(self.pwNet:get(1):get(1):get(1).weight[-1]:abs():sum() == 0)
    
    for d = 1, pwDevData.numDocs do
      if d % 100 == 0 then
        print("dev doc " .. tostring(d))
        collectgarbage()
      end
      local numMents = anaDevData:numMents(d)
      ofi:write("0") -- predict first thing links to itself always 
      
      local pwDocBatch = pwDevData:getDocBatch(d)
      local anaDocBatch = anaDevData:docBatch(d) -- i know, i know...
      if cuda then
        pwDocBatch = pwDocBatch:cuda()
        anaDocBatch = anaDocBatch:cuda()
      end

      for m = 2, numMents do
        local start = ((m-2)*(m-1))/2 -- one behind first pair for mention m

        local scores = self.pwNet:forward({pwDocBatch:sub(start+1,start+m-1),
            anaDocBatch:sub(m,m):expand(m-1,anaDocBatch:size(2))}):squeeze(2)
        local naScore = self.naNet:forward(anaDocBatch:sub(m,m)):squeeze()
      
        local highAntScore, pred = torch.max(scores,1) -- only considers antecedents
        if naScore > highAntScore[1] then
          pred[1] = m
        end
        ofi:write(" ",tostring(pred[1]-1))
      end 
      ofi:write("\n")
    end
    ofi:close()
  end  

end


function predictThings(pwNetFi,naNetFi,lstmFi,pwDevData,anaDevData,cuda,bpfi)
  local model = SavedVanillaMR(naNetFi, pwNetFi, cuda)
  model.naNet:evaluate()
  model.pwNet:evaluate() 
  model:writeBPs(pwDevData,anaDevData,cuda,bpfi)
end


function train(pwData,anaData,trOPCs,cdb,pwDevData,anaDevData,devOPCs,devCdb,Hp,Ha,Hs,H2,
              fl,fn,wl,nEpochs,save,savePfx,cuda,anteSerFi,anaSerFi) 
  local serFi = string.format("models/%s-vanilla-%d-%d.model", savePfx, Hp, Ha)
  local eta0 = 1e-1
  local eta1 = 2e-3
  local model = VanillaMR(pwData.maxFeat+1, Hp, anaData.maxFeat+1, Ha, fl, fn, wl,
    cuda,anteSerFi, anaSerFi, opts.dop)
  local statez = {{},{},{},{},{},{},{},{}}
  local deltTensor = cuda and torch.ones(1,1):cuda() or torch.ones(1,1)
  local keepGoing = true
  local ep = 1
  while keepGoing do
    print("epoch: " .. tostring(ep))
    model.naNet:training()
    model.pwNet:training()
    -- use document sized minibatches
    for d = 1, pwData.numDocs do
     -- print(d)
      if d % 200 == 0 then
        print("doc " .. tostring(d))
        collectgarbage()
      end 
      local pwDocBatch = pwData:getDocBatch(d)
      local anaDocBatch = anaData:docBatch(d) -- i know, i know...
      if cuda then
        pwDocBatch = pwDocBatch:cuda()
        anaDocBatch = anaDocBatch:cuda()
      end
      model.pwNet:zeroGradParameters()
      model.naNet:zeroGradParameters()
      
      model:docGrad(d,pwDocBatch,anaDocBatch,trOPCs[d],deltTensor,anaData:numMents(d),cdb)
      
      mu.adagradStep(model.naNet:get(1).weight,
                     model.naNet:get(1).gradWeight,eta0,statez[1])
      mu.adagradStep(model.naNet:get(3).bias,
                     model.naNet:get(3).gradBias,eta0,statez[2])
                      
      mu.adagradStep(model.pwNet:get(1):get(1):get(1).weight,
                     model.pwNet:get(1):get(1):get(1).gradWeight,eta0,statez[3])
      mu.adagradStep(model.pwNet:get(1):get(1):get(3).bias,
                     model.pwNet:get(1):get(1):get(3).gradBias,eta0,statez[4])
                      
      mu.adagradStep(model.naNet:get(5).weight,
                     model.naNet:get(5).gradWeight,eta1,statez[5])
      mu.adagradStep(model.naNet:get(5).bias,
                     model.naNet:get(5).gradBias,eta1,statez[6])
                      
      mu.adagradStep(model.pwNet:get(4).weight,
                     model.pwNet:get(4).gradWeight,eta1,statez[7])
      mu.adagradStep(model.pwNet:get(4).bias,
                     model.pwNet:get(4).gradBias,eta1,statez[8])              
    end
    if save then
      print("overwriting params...")
      torch.save(serFi.."-vanilla-pw", model.pwNet)
      torch.save(serFi.."-vanilla-na", model.naNet)
    end                  
    collectgarbage()
    if ep >= nEpochs then
      keepGoing = false
    end    
    ep = ep + 1
  end -- end while keepGoing
end

-------------------------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training vanilla mention ranking model')
cmd:text()
cmd:text('Options')
cmd:option('-Hp', 700, 'Pairwise network hidden layer size')
cmd:option('-Ha', 200, 'Anaphoricity network hidden layer size')
cmd:option('-trainClustFile', '../SMALLTrainOPCs.txt', 'Train Oracle Predicted Clustering File')
cmd:option('-devClustFile', '../SMALLDevOPCs.txt', 'Train Oracle Predicted Clustering File')
cmd:option('-pwTrFeatPrefix', 'train_small', 'Expects train pairwise features in <pwTrFeatPfx>-pw-*.h5')
cmd:option('-pwDevFeatPrefix', 'dev_small', 'Expects dev pairwise features in <pwDevFeatPfx>-pw-*.h5')
cmd:option('-anaTrFeatPrefix', 'train_small', 'Expects train anaphoricity features in <anaTrFeatPfx>-na-*.h5')
cmd:option('-anaDevFeatPrefix', 'dev_small', 'Expects dev anaphoricity features in <anaDevFeatPfx>-na-*.h5')
cmd:option('-nEpochs', 5, 'Max number of epochs to train')
cmd:option('-fl', 0.5, 'False Link cost')
cmd:option('-fn', 1.2, 'False New cost')
cmd:option('-wl', 1, 'Wrong Link cost')
cmd:option('-dop', 0.4, 'Dropout rate on pairwise scorer')
cmd:option('-gpuid', -1, 'Gpu ID (leave -1 for cpu)')
cmd:option('-save', false, 'Save model')
cmd:option('-savePfx', 'simple', 'Prefixes saved model with this')
cmd:option('-anteSerFi', 'models/small_700.model-pw-0.100000', 'Serialized pre-trained antecedent ranking model')
cmd:option('-anaSerFi', 'models/small_200.model-na-0.100000', 'Serialized pre-trained anaphoricity model')
cmd:option('-PT', false, ' pretrain')
cmd:option('-loadAndPredict', false, 'Load full model and predict (on dev or test)')
cmd:option('-savedPWNetFi', '', 'Saved pairwise network model file (for prediction)')
cmd:option('-savedNANetFi', '', 'Saved NA network model file (for prediction)')
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
  if not opts.loadAndPredict then -- if training, get train data
    local pwTrData = SpDMPWData.loadFromH5(opts.pwTrFeatPrefix)
    print("read pw train data")
    print("max pw feature is: " .. pwTrData.maxFeat)
    local anaTrData = SpKFDMData(opts.anaTrFeatPrefix)
    print("read anaph train data")
    print("max ana feature is: " .. anaTrData.maxFeat)       
    local trOPCs = getOPCs(opts.trainClustFile,anaTrData) 
    print("read train clusters")    
    local pwDevData = SpDMPWData.loadFromH5(opts.pwDevFeatPrefix)
    print("read pw dev data")
    local anaDevData = SpKFDMData(opts.anaDevFeatPrefix)
    print("read anaph dev data")  
    local devOPCs = getOPCs(opts.devClustFile,anaDevData) 
    print("read dev clusters")          
    if opts.gpuid >= 0 then
      for i = 1, #trOPCs do
        trOPCs[i]:cudify()
      end
      for i = 1, #devOPCs do
        devOPCs[i]:cudify()
      end
    end   
    print(opts)
    train(pwTrData,anaTrData,trOPCs,cdb,pwDevData,anaDevData,devOPCs,devCdb,opts.Hp,opts.Ha,
      opts.Hs,opts.H2,opts.fl,opts.fn,opts.wl,opts.nEpochs,opts.save,opts.savePfx,opts.gpuid >= 0,
      opts.PT and opts.anteSerFi or false, opts.PT and opts.anaSerFi or false) 
  else
    require 'cutorch'
    require 'cunn'    
    local pwDevData = SpDMPWData.loadFromH5(opts.pwDevFeatPrefix)
    print("read pw dev data")
    local anaDevData = SpKFDMData(opts.anaDevFeatPrefix)
    print("read anaph dev data")  
    local pwNetFi = opts.savedPWNetFi
    local naNetFi = opts.savedNANetFi
    print(opts)
    local bpfi = "bps/" .. tostring(os.time()) .. "dev.bps"
    print("using bpfi: " .. bpfi)
    predictThings(pwNetFi,naNetFi,lstmFi,pwDevData,anaDevData,opts.gpuid >= 0,bpfi)
  end
end

main()

