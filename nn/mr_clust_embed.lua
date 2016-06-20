require 'nn'
require 'rnn'
require 'coref_utils'
require 'sparse_doc_data'
require 'clust_batcher'

local mu = require 'model_utils'

do
  local BaseClustModel = torch.class('BaseClustModel')

  function BaseClustModel:docClustStuff(d,docBatch,cdb,train,ldop)
    local clustModels = {}
    local clustStates = {}
    local clustBatches = {}
    local clustGOs = {} -- grad outputs
    local nextGOStart = 1
    for size, clusts in pairs(cdb.docIdxs[d]) do 
      local clustBatch = docBatch:index(1,clusts:view(-1))
      clustBatches[size] =  clustBatch
      -- lazily instantiate clones
      if self.clustModClones[size] == nil then
        local clustModel = self.pwNet:get(1):get(2):get(1):get(1):clone('weight','gradWeight',
          'bias','gradBias')
        clustModel:add(nn.View(-1,size,self.Hs))
        clustModel:add(nn.SplitTable(2,3))
        clustModel:add(self.lstm:clone('weight','gradWeight','bias','gradBias'))
        clustModel:add(nn.Sequencer(nn.Dropout(ldop)))
        self.clustModClones[size] = self.cuda and clustModel:cuda() or clustModel      
        self.clustModClones[size]:get(1):get(1):share(self.pwNet:get(1):get(2):get(1):get(1
            ):get(1):get(1), 'weight', 'gradWeight', 'bias', 'gradBias')
        self.clustModClones[size]:get(1):get(2):share(self.pwNet:get(1):get(2):get(1):get(1
            ):get(1):get(2),'weight', 'gradWeight', 'bias', 'gradBias')
        self.clustModClones[size]:get(7):share(self.lstm,'weight','gradWeight','bias','gradBias')
      end
      local cm = self.clustModClones[size]
      cm:get(7):forget() -- just in case...
      if train then
        cm:training()
      else
        cm:evaluate()
      end
      -- should produce table of length clusts:size(2) containing clusts:size(1) x H tensors 
      -- so index with [size][step][row]
      clustStates[size] = cm:forward({clustBatch, cdb.docPosns[d][size]}) 
      clustModels[size] = cm
      -- also index clustGOs with [size][step][row]
      clustGOs[size] = self.clustGOs:sub(nextGOStart,nextGOStart+
        (clusts:size(2)*clusts:size(1)*self.Hs)-1):view(clusts:size(2),clusts:size(1),self.Hs):zero()  
      nextGOStart = nextGOStart + clusts:size(2)*clusts:size(1)*self.Hs
    end
    return clustModels, clustStates, clustBatches, clustGOs
  end
  
  function BaseClustModel:getDocPosns(numMents)
    local docPosns = self.docPosns:sub(1,numMents):copy(self.range:sub(1,numMents)):view(numMents,1)
    docPosns:mul(2)
    docPosns:add(-numMents-1)
    docPosns:div(numMents-1)
    return docPosns
  end
  
end

do

  local ClustDotModel, parent = torch.class('ClustDotModel', 'BaseClustModel')

  function ClustDotModel:__init(pwD, Hp, anaD, Ha, Hs, maxNumMents, maxNumClusts,
      fl, fn, wl, cuda, anteSerFi, anaSerFi, dop) 
    torch.manualSeed(2)
    if cuda then
      cutorch.manualSeed(2)
    end
     
    self.cuda = cuda
    self.fl = fl
    self.fn = fn
    self.wl = wl
    self.Hs = Hs

    local naNet = mu.make_sp_mlp(anaD,Ha,true,false)
    if anaSerFi then
      print("preinit'ing naNet with " .. anaSerFi)
      local preNaNet = torch.load(anaSerFi)
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
    
    local clustNAMLP = nn.Sequential():add(nn.ParallelTable():add(
                            nn.Sequential():add(nn.LookupTable(anaD,Ha)):add(nn.Sum(2))):add(
                            nn.Sequential():add(nn.Sum(1)):add(nn.View(1,-1)):add(nn.Linear(Hs,Ha)))):add(
                                        nn.CAddTable()):add(nn.Tanh()):add(nn.Dropout(opts.nadop)):add(
                                        nn.Linear(Ha,1))
    recSutsInit(clustNAMLP,15) -- init weights sparsely
    local naScorer = nn.Sequential():add(nn.ParallelTable():add(naNet):add(clustNAMLP)):add(nn.CAddTable())

    -- make net for scoring anaphoric case
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

    -- add clone of naNet
    local naNetClone = naNet:clone('weight','bias','gradWeight','gradBias')
    -- get rid of dropout and final linear
    naNetClone.modules[5] = nil
    naNetClone.modules[6] = nil
    collectgarbage()

    local origScorer = nn.Sequential():add(nn.ParallelTable():add(anteNet):add(naNetClone))
    origScorer:add(nn.JoinTable(2,2))
    origScorer:add(nn.Dropout(opts.fdop))
    origScorer:add(nn.Linear(Hp+Ha,1))
    
    -- maps features into a space useful for cluster stuff
    local clustEmbedder = mu.make_sp_and_dense_mlp(anaD,1,Hs,true,true)
    clustEmbedder:add(nn.Dropout(opts.cmdop))
    local structScorer = nn.Sequential():add(nn.ParallelTable():add(clustEmbedder):add(nn.Identity()))
    structScorer:add(nn.DotProduct())
    
    local firstLayer = nn.ParallelTable() -- joins all reps
    firstLayer:add(origScorer)    
    firstLayer:add(structScorer)

    local pwNet = nn.Sequential():add(firstLayer):add(nn.CAddTable())
    
    local lstm = nn.Sequencer(nn.FastLSTM(Hs,Hs))
    lstm:remember('neither')
    -- init LSTM weights
    local unif = 0.05
    lstm.module.i2g.weight:uniform(-unif,unif)
    lstm.module.o2g.weight:uniform(-unif,unif)
    lstm.module.i2g.bias:uniform(-unif,unif) 
    lstm.module.i2g.bias:sub(Hs*2+1,Hs*3):fill(1)
    
    -- set dummy feature weigts to zero
    clustNAMLP:get(1):get(1):get(1).weight[-1]:fill(0)    
    naNet:get(1).weight[-1]:fill(0)
    anteNet:get(1).weight[-1]:fill(0)
    clustEmbedder:get(1):get(1):get(1).weight[-1]:fill(0)
    
    self.naScorer = cuda and naScorer:cuda() or naScorer
    self.pwNet = cuda and pwNet:cuda() or pwNet
    self.lstm = cuda and lstm:cuda() or lstm
    self.clustModClones = {}
    
    -- may need to reshare anaphoricity feature weights (if cuda'ing)
    self.pwNet:get(1):get(1):get(1):get(2):get(1):share(self.naScorer:get(1):get(1):get(1), 'weight', 'gradWeight')
    self.pwNet:get(1):get(1):get(1):get(2):get(3):share(self.naScorer:get(1):get(1):get(3), 'bias', 'gradBias')
    
    self.range = cuda and torch.range(1,maxNumMents):cuda() or torch.range(1,maxNumMents)
    self.docPosns = cuda and torch.zeros(maxNumMents):cuda() or torch.zeros(maxNumMents)
    self.clustStates = cuda and torch.zeros(maxNumClusts,Hs):cuda() or torch.zeros(maxNumClusts,Hs)
    self.antClustStates = cuda and torch.zeros(maxNumMents,Hs):cuda() or torch.zeros(maxNumMents,Hs)
    self.clustGOs = cuda and torch.zeros(maxNumMents*Hs):cuda() or torch.zeros(maxNumMents*Hs)  
    collectgarbage()
  end 
  
  function ClustDotModel:docGrad(d,pwDocBatch,anaDocBatch,OPC,deltTensor,numMents,cdb,ldop)
    -- pre-run everything thru lstms...
    local clustModels, clustStates, clustBatches, clustGOs = self:docClustStuff(d,anaDocBatch,cdb,
      true,ldop)
    -- get dense distance stuff
    local docPosns = self:getDocPosns(numMents) 
    local nextIdxs = torch.Tensor(#OPC.clusts):fill(2) -- index of next state for each cluster
    local docLocs = cdb.docLocs[d]
    -- index clustStates with [size][step][row]
    self.clustStates[1] = clustStates[docLocs[1][1]][1][docLocs[1][2]] -- last state of each cluster
    self.antClustStates[1] = self.clustStates[1] -- last state of the cluster each antecedent is in
    local currNumClusts = 1
    
    for m = 2, numMents do -- ignore first guy; always NA 
      local cid = OPC.m2c[m]
      local cidSize = docLocs[cid][1]
      local cidRow = docLocs[cid][2]
      local start = ((m-2)*(m-1))/2 -- one behind first pair for mention m
      
      local scores = self.pwNet:forward({
          { pwDocBatch:sub(start+1,start+m-1),anaDocBatch:sub(m,m):expand(m-1,anaDocBatch:size(2)) },
          { { anaDocBatch:sub(m,m):expand(m-1,anaDocBatch:size(2)),docPosns:sub(m,m):expand(m-1,1) },
            self.antClustStates:sub(1,m-1) } }):squeeze(2)
      
      local naScore = self.naScorer:forward({anaDocBatch:sub(m,m),
          {anaDocBatch:sub(m,m),self.clustStates:sub(1,currNumClusts)}}):squeeze()

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
          self.pwNet:forward({ { pwDocBatch:sub(start+pred,start+pred), anaDocBatch:sub(m,m) },
              { { anaDocBatch:sub(m,m), docPosns:sub(m,m) }, self.antClustStates:sub(pred,pred) } })
          local lstmGIs = self.pwNet:backward({ { pwDocBatch:sub(start+pred,start+pred), 
                anaDocBatch:sub(m,m) }, { { anaDocBatch:sub(m,m), docPosns:sub(m,m) }, 
                self.antClustStates:sub(pred,pred) } }, deltTensor)          

          -- accumulate gradients wrt LSTM
          local predCid = OPC.m2c[pred]
          local psize = docLocs[predCid][1]
          local prow = docLocs[predCid][2]
          clustGOs[psize][nextIdxs[predCid]-1][prow]:add(lstmGIs[2][2])         
        else
          -- the predicted thing must have been m, which we already ran thru naNet
          local naLstmGIs = self.naScorer:backward({anaDocBatch:sub(m,m),
          {anaDocBatch:sub(m,m),self.clustStates:sub(1,currNumClusts)}},deltTensor)
          -- need to add gradients to all current LSTM states
          for c = 1, currNumClusts do
            local csize,crow = docLocs[c][1], docLocs[c][2]
            clustGOs[csize][nextIdxs[c]-1][crow]:add(naLstmGIs[2][2][c])
          end
        end
        -- same deal for subtracting latent thing
        if late ~= m then          
          self.pwNet:forward({ { pwDocBatch:sub(start+late,start+late), anaDocBatch:sub(m,m) },
              { { anaDocBatch:sub(m,m),docPosns:sub(m,m) }, self.antClustStates:sub(late,late) } })
          local lstmGIs = self.pwNet:backward({ { pwDocBatch:sub(start+late,start+late), 
                anaDocBatch:sub(m,m) }, { { anaDocBatch:sub(m,m),docPosns:sub(m,m) }, 
                self.antClustStates:sub(late,late) } }, -deltTensor) 
          local lateCid = OPC.m2c[late]
          local lsize = docLocs[lateCid][1]
          local lrow = docLocs[lateCid][2]
          clustGOs[lsize][nextIdxs[lateCid]-1][lrow]:add(lstmGIs[2][2])
        else
          local naLstmGIs = self.naScorer:backward({anaDocBatch:sub(m,m),
          {anaDocBatch:sub(m,m),self.clustStates:sub(1,currNumClusts)}},-deltTensor)
          -- need to add gradients to all current LSTM states
          for c = 1, currNumClusts do
            local csize,crow = docLocs[c][1], docLocs[c][2]
            clustGOs[csize][nextIdxs[c]-1][crow]:add(naLstmGIs[2][2][c])
          end
        end
      end  -- end if delt > 0
      if OPC:anaphoric(m) then -- update currStateIdx
        -- update Hs and nextIdxs
        self.clustStates[cid] = clustStates[cidSize][nextIdxs[cid]][cidRow]
        -- now need to assign this most recent state to all antecedents in this cluster
        -- (indexCopy() seems very slightly faster than using a loop, even tho it does extra work)
        self.antClustStates:indexCopy(1,OPC.clusts[cid],
          self.clustStates:sub(cid,cid):expand(OPC.clusts[cid]:size(1),self.clustStates:size(2)))
        nextIdxs[cid] = nextIdxs[cid] + 1
      else
        -- update Hs and nextIdxs
        self.clustStates[cid] = clustStates[cidSize][1][cidRow]
        -- if not anaphoric, mention m's current state is just cid's state
        self.antClustStates[m] = self.clustStates[cid]
        currNumClusts = currNumClusts + 1        
      end
    end -- end for m
    -- now that we've accummulated all lstm gradient outputs, we BPTT
    for size,lstm in pairs(clustModels) do
      lstm:backward({clustBatches[size], cdb.docPosns[d][size]}, torch.split(clustGOs[size],1))
    end
  end  

end

do
  local SavedClustDotModel, parent = torch.class('SavedClustDotModel', 'BaseClustModel')

  function SavedClustDotModel:__init(maxNumMents, maxNumClusts, pwNetFi, naNetFi, lstmFi, cuda) 
    torch.manualSeed(2)
    if cuda then
      cutorch.manualSeed(2)
    end
    self.cuda = cuda
    self.naScorer = cuda and torch.load(naNetFi) or torch.load(naNetFi):float()
    self.pwNet = cuda and torch.load(pwNetFi) or torch.load(pwNetFi):float()
    self.lstm = cuda and torch.load(lstmFi) or torch.load(lstmFi):float()
    local Hs = self.lstm:get(1).i2g.weight:size(2) 
    self.Hs = Hs
    self.clustModClones = {}
    
    self.range = cuda and torch.range(1,maxNumMents):cuda() or torch.range(1,maxNumMents):float()
    self.docPosns = cuda and torch.zeros(maxNumMents):cuda() or torch.zeros(maxNumMents):float()
    self.clustStates = cuda and torch.zeros(maxNumClusts,Hs):cuda() or torch.zeros(maxNumClusts,Hs):float()
    self.antClustStates = cuda and torch.zeros(maxNumMents,Hs):cuda() or torch.zeros(maxNumMents,Hs):float()
    self.clustGOs = cuda and torch.zeros(maxNumMents*Hs):cuda() or torch.zeros(maxNumMents*Hs):float()
    collectgarbage()
  end 


  function SavedClustDotModel:writeGreedyBPs(pwDevData,anaDevData,cuda,bpfi)
    local bpfi = bpfi or "bps/dev.bps"
    local ofi = assert(io.open(bpfi,"w"))
    assert(self.naScorer.train == false)
    assert(self.pwNet.train == false)
    assert(self.lstm.train == false)
    
    -- just cloning the embedder b/c we'll be reusing pwNet below...
    local embedder = self.pwNet:get(1):get(2):get(1):get(1):clone('weight','gradWeight','bias','gradBias')    
    local lstm = self.lstm:get(1).recurrentModule
    local tout,tcell
    local Hs = self.lstm:get(1).i2g.weight:size(2)
    if not self.tzero then
      self.tzero = torch.Tensor.new():typeAs(self.lstm:get(1).i2g.weight):resize(Hs)
    end
    local tzero = self.tzero
    tzero:zero()
    local startLstm = lstm:clone('weight','gradWeight','bias','gradBias')
    
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

      local docPosns = self:getDocPosns(numMents)
      local Inp = embedder:forward({anaDocBatch,docPosns})
      -- this will do extra work but seems to be slightly faster
      local StartOuts, StartCells = unpack(startLstm:forward({Inp,tzero:view(1,Hs):expand(numMents,Hs),tzero:view(1,Hs):expand(numMents,Hs)}))
      
      tout, tcell = StartOuts[1], StartCells[1]
      self.clustStates[1] = tout
      self.antClustStates[1] = self.clustStates[1]
      -- we use the following to store running predicted cluster info
      local predCids = torch.ones(numMents)
      local predClusts = {{1}}
      local cellOuts = {{}}
      cellOuts[1].cell, cellOuts[1].output = tcell, tout
      local currNumClusts = 1
            
      for m = 2, numMents do
        local start = ((m-2)*(m-1))/2 -- one behind first pair for mention m
        local scores = self.pwNet:forward({ { pwDocBatch:sub(start+1,start+m-1),
            anaDocBatch:sub(m,m):expand(m-1,anaDocBatch:size(2)) },
            { { anaDocBatch:sub(m,m):expand(m-1,anaDocBatch:size(2)),docPosns:sub(m,m):expand(m-1,1) },
            self.antClustStates:sub(1,m-1) } }):squeeze(2)
        local naScore = self.naScorer:forward({anaDocBatch:sub(m,m),
          {anaDocBatch:sub(m,m),self.clustStates:sub(1,currNumClusts)}}):squeeze()
      
        local highAntScore, pred = torch.max(scores,1) -- only considers antecedents
        if naScore > highAntScore[1] then
          pred[1] = m
        end
        
        ofi:write(" ",tostring(pred[1]-1))
        if pred[1] ~= m then -- predicted anaphoric
          local predCid = predCids[pred[1]]
          -- update our running clusters with this prediction
          predCids[m] = predCid          
          table.insert(predClusts[predCid],m)
          -- get current state of this cluster
	        tout, tcell = unpack(lstm:forward({Inp[m],cellOuts[predCid].output,cellOuts[predCid].cell}))
	        self.clustStates[predCid] = tout
          -- now copy to all antecedents in the cluster 
          self.antClustStates:indexCopy(1,torch.LongTensor(predClusts[predCid]),
            self.clustStates:sub(predCid,predCid):expand(#predClusts[predCid],self.clustStates:size(2)))
          -- update with latest cell and output
          cellOuts[predCid].cell, cellOuts[predCid].output = tcell:clone(), tout:clone()
        else -- predicted non-anaphoric, so starting a new cluster
          currNumClusts = currNumClusts + 1
          local newCid  = currNumClusts
          predCids[m] = newCid
          predClusts[newCid] = {m}
          tout, tcell = StartOuts[m], StartCells[m]
          self.clustStates[newCid] = tout
          cellOuts[newCid] = {}
          cellOuts[newCid].cell, cellOuts[newCid].output = tcell, tout
          self.antClustStates[m] = self.clustStates[newCid]
        end
      end 
      ofi:write("\n")
      collectgarbage()
    end
    ofi:close()
  end
end


function train(pwTrData,anaTrData,trOPCs,cdb,pwDevData,anaDevData,devOPCs,devCdb,Hp,Ha,Hs,
              fl,fn,wl,nEpochs,save,savePfx,cuda,bpfi,anteSerFi,anaSerFi)
  local maxNumMents, maxNumClusts = getMaxes(anaTrData,anaDevData,trOPCs,devOPCs)
  print(string.format("maxNumMents = %d, maxNumClusts = %d", maxNumMents, maxNumClusts))  
  local serFi = string.format("models/%s-mce-%d-%d.model", savePfx, Hp, Hs)
  local eta0 = 1e-1
  local eta1 = 2e-3
  local eta2 = 1e-2
  local eta3 = 2e-3
  local dop = 0.4
  local bestF = 0
  local pwDatas = {pwTrData, pwDevData}
  local anaDatas = {anaTrData, anaDevData}
  local OPCs = {trOPCs, devOPCs}
  local cdbs = {cdb, devCdb}
  print(string.format("doing %f %f %f %f %f", eta0, eta1,eta2,eta3,dop))
  local model = ClustDotModel(pwTrData.maxFeat+1, Hp, anaTrData.maxFeat+1, Ha, Hs,
    maxNumMents, maxNumClusts, fl, fn, wl,cuda,anteSerFi, anaSerFi, dop)
  local statez = {{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}} 
  local deltTensor = cuda and torch.ones(1,1):cuda() or torch.ones(1,1)
  local keepGoing = true
  local ep = 1
  while keepGoing do
    print("epoch: " .. tostring(ep))
    model.naScorer:training()
    model.pwNet:training()
    model.lstm:training()
    for qq = 1, #pwDatas do
      local pwData = pwDatas[qq]
      local anaData = anaDatas[qq]
    -- use document sized minibatches
      for d = 1, pwData.numDocs do
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
        model.naScorer:zeroGradParameters()
        model.pwNet:zeroGradParameters()
        model.lstm:zeroGradParameters()
        model:docGrad(d,pwDocBatch,anaDocBatch,OPCs[qq][d],deltTensor,anaData:numMents(d),cdbs[qq],opts.ldop)
        -- clip gradients
        model.lstm:get(1).i2g.gradWeight:clamp(-opts.clip, opts.clip)
        model.lstm:get(1).i2g.gradBias:clamp(-opts.clip, opts.clip)
        model.lstm:get(1).o2g.gradWeight:clamp(-opts.clip, opts.clip)
        model.pwNet:get(1):get(2):get(1):get(1):get(1):get(1):get(1).gradWeight:clamp(-opts.clip, opts.clip)
        model.pwNet:get(1):get(2):get(1):get(1):get(1):get(2):get(1).gradWeight:clamp(-opts.clip, opts.clip)
        model.pwNet:get(1):get(2):get(1):get(1):get(1):get(2):get(1).gradBias:clamp(-opts.clip, opts.clip)

        -- TODO: make this less awful
        mu.adagradStep(model.pwNet:get(1):get(1):get(1):get(1):get(1).weight,
                       model.pwNet:get(1):get(1):get(1):get(1):get(1).gradWeight,eta0,statez[1])
        mu.adagradStep(model.pwNet:get(1):get(1):get(1):get(1):get(3).bias,
                       model.pwNet:get(1):get(1):get(1):get(1):get(3).gradBias,eta0,statez[2])

        mu.adagradStep(model.naScorer:get(1):get(1):get(1).weight,
                       model.naScorer:get(1):get(1):get(1).gradWeight,eta0,statez[3])
        mu.adagradStep(model.naScorer:get(1):get(1):get(3).bias,
                       model.naScorer:get(1):get(1):get(3).gradBias,eta0,statez[4])

        mu.adagradStep(model.pwNet:get(1):get(1):get(4).weight,
                       model.pwNet:get(1):get(1):get(4).gradWeight,eta1,statez[5])
        mu.adagradStep(model.pwNet:get(1):get(1):get(4).bias,
                       model.pwNet:get(1):get(1):get(4).gradBias,eta1,statez[6])
                     
        mu.adagradStep(model.naScorer:get(1):get(1):get(5).weight,
                       model.naScorer:get(1):get(1):get(5).gradWeight,eta1,statez[7])
        mu.adagradStep(model.naScorer:get(1):get(1):get(5).bias,
                       model.naScorer:get(1):get(1):get(5).gradBias,eta1,statez[8])                                 
        
        mu.adagradStep(model.pwNet:get(1):get(2):get(1):get(1):get(1):get(1):get(1).weight,
                       model.pwNet:get(1):get(2):get(1):get(1):get(1):get(1):get(1).gradWeight,
                       eta2,statez[9])
        mu.adagradStep(model.pwNet:get(1):get(2):get(1):get(1):get(1):get(2):get(1).weight,
                       model.pwNet:get(1):get(2):get(1):get(1):get(1):get(2):get(1).gradWeight,
                       eta2,statez[10])
        mu.adagradStep(model.pwNet:get(1):get(2):get(1):get(1):get(1):get(2):get(1).bias,
                       model.pwNet:get(1):get(2):get(1):get(1):get(1):get(2):get(1).gradBias,
                       eta2,statez[11])
        
        mu.adagradStep(model.lstm:get(1).i2g.weight,
                       model.lstm:get(1).i2g.gradWeight,eta3,statez[12])
        mu.adagradStep(model.lstm:get(1).i2g.bias,
                       model.lstm:get(1).i2g.gradBias,eta3,statez[13])
        mu.adagradStep(model.lstm:get(1).o2g.weight,
                       model.lstm:get(1).o2g.gradWeight,eta3,statez[14])
        
        mu.adagradStep(model.naScorer:get(1):get(2):get(1):get(1):get(1).weight,
                       model.naScorer:get(1):get(2):get(1):get(1):get(1).gradWeight,
                       eta2,statez[15])
        mu.adagradStep(model.naScorer:get(1):get(2):get(1):get(2):get(3).weight,
                       model.naScorer:get(1):get(2):get(1):get(2):get(3).gradWeight,
                       eta2,statez[16])
        mu.adagradStep(model.naScorer:get(1):get(2):get(1):get(2):get(3).bias,
                       model.naScorer:get(1):get(2):get(1):get(2):get(3).gradBias,
                       eta2,statez[17])   
                     
        mu.adagradStep(model.naScorer:get(1):get(2):get(5).weight,
                       model.naScorer:get(1):get(2):get(5).gradWeight,eta1,statez[18])
        mu.adagradStep(model.naScorer:get(1):get(2):get(5).bias,
                       model.naScorer:get(1):get(2):get(5).gradBias,eta1,statez[19])                                 
      end
    end
    model.naScorer:evaluate()
    model.pwNet:evaluate()
    model.lstm:evaluate()  
    if save then
      print("overwriting params...")
      torch.save(serFi.."-na", model.naScorer)
      torch.save(serFi.."-pw", model.pwNet)
      torch.save(serFi.."-lstm",model.lstm)
    end                  
    collectgarbage()
    if ep >= nEpochs then
      keepGoing = false
    end
    ep = ep + 1
  end -- end while keepGoing
end


function predictThings(pwNetFi,naNetFi,lstmFi,pwDevData,anaDevData,cuda,bpfi)
  local maxNumMents, maxNumClusts = 1296, 896 -- these are just the numbers from train
  local model = SavedClustDotModel(maxNumMents, maxNumClusts, pwNetFi, naNetFi, lstmFi, cuda)
  model.naScorer:evaluate()
  model.pwNet:evaluate()
  model.lstm:evaluate()  
  collectgarbage()
  model:writeGreedyBPs(pwDevData,anaDevData,cuda,bpfi)
end

-------------------------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training mention ranking + cluster embedding model')
cmd:text()
cmd:text('Options')
cmd:option('-Hp', 700, 'Pairwise network hidden layer size')
cmd:option('-Ha', 200, 'Anaphoricity network hidden layer size')
cmd:option('-Hs', 200, 'Cluster embedding size')
cmd:option('-trainClustFile', '../TrainOPCs.txt', 'Train Oracle Predicted Clustering File')
cmd:option('-devClustFile', '../DevOPCs.txt', 'Train Oracle Predicted Clustering File')
cmd:option('-pwTrFeatPrefix', 'train_small', 'Expects train pairwise features in <pwTrFeatPfx>-pw-*.h5')
cmd:option('-pwDevFeatPrefix', 'dev_small', 'Expects dev pairwise features in <pwDevFeatPfx>-pw-*.h5')
cmd:option('-anaTrFeatPrefix', 'train_small', 'Expects train anaphoricity features in <anaTrFeatPfx>-na-*.h5')
cmd:option('-anaDevFeatPrefix', 'dev_small', 'Expects dev anaphoricity features in <anaDevFeatPfx>-na-*.h5')
cmd:option('-nEpochs', 5, 'Max number of epochs to train')
cmd:option('-fl', 0.5, 'False Link cost')
cmd:option('-fn', 1.2, 'False New cost')
cmd:option('-wl', 1, 'Wrong Link cost')
cmd:option('-gpuid', -1, 'Gpu id (leave -1 for cpu)')
cmd:option('-save', false, 'Save model')
cmd:option('-savePfx', 'simple', 'Prefixes saved model with this')
cmd:option('-anteSerFi', 'models/small_700.model-pw-0.100000', 'serialized pre-trained antecedent ranking model')
cmd:option('-anaSerFi', 'models/small_200.model-na-0.100000', 'serialized pre-training anaphoricity model')
cmd:option('-PT', false, 'pretrain')
cmd:option('-fdop', 0.4, 'dropout rate on local, f scorer')
cmd:option('-cmdop', 0, 'dropout rate on mention embeddings')
cmd:option('-nadop', 0, 'dropout rate on (cluster) NA MLP')
cmd:option('-ldop', 0.3, 'dropout rate on lstm output')
cmd:option('-clip', 10, 'clip gradients to lie in (-clip, clip)')
cmd:option('-loadAndPredict', false, 'Load full model and predict (on dev or test)')
cmd:option('-savedPWNetFi', '', 'trained pairwise network')
cmd:option('-savedNANetFi', '', 'trained NA network')
cmd:option('-savedLSTMFi', '', 'trained lstm')
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
    local trOPCs = getOPCs(opts.trainClustFile,anaTrData) -- torch.load('trOPCs.dat') --
    print("read train clusters")    
    local pwDevData = SpDMPWData.loadFromH5(opts.pwDevFeatPrefix)
    print("read pw dev data")
    local anaDevData = SpKFDMData(opts.anaDevFeatPrefix)
    print("read anaph dev data")  
    local devOPCs =  getOPCs(opts.devClustFile,anaDevData) -- torch.load('devOPCs.dat')
    print("read dev clusters")          
    local cdb =  ClustAndDistBatcher(trOPCs,anaTrData) -- torch.load('cdb.dat')
    print("made train clust batcher")
    local devCdb = ClustAndDistBatcher(devOPCs,anaDevData) -- torch.load('devCdb.dat') 
    print("made dev clust batcher")
    if opts.gpuid >= 0 then
      cdb:cudify()
      for i = 1, #trOPCs do
        trOPCs[i]:cudify()
      end
      devCdb:cudify()
      for i = 1, #devOPCs do
        devOPCs[i]:cudify()
      end
    end   
    print(opts)
    local bpfi = "bps/" .. tostring(os.time()) .. "dev.bps"
    print("using bpfi: " .. bpfi)
    train(pwTrData,anaTrData,trOPCs,cdb,pwDevData,anaDevData,devOPCs,devCdb,opts.Hp,opts.Ha,
      opts.Hs,opts.fl,opts.fn,opts.wl,opts.nEpochs,opts.save,opts.savePfx,opts.gpuid >= 0,
      bpfi,opts.PT and opts.anteSerFi or false, opts.PT and opts.anaSerFi or false)
       
  else
    require 'cutorch'
    require 'cunn'    
    local pwDevData = SpDMPWData.loadFromH5(opts.pwDevFeatPrefix)
    print("read pw dev data")
    local anaDevData = SpKFDMData(opts.anaDevFeatPrefix)
    local pwNetFi = opts.savedPWNetFi
    local naNetFi = opts.savedNANetFi
    local lstmFi = opts.savedLSTMFi
    print(opts)
    local bpfi = "bps/" .. tostring(os.time()) .. "dev.bps"
    print("using bpfi: " .. bpfi)
    predictThings(pwNetFi,naNetFi,lstmFi,pwDevData,anaDevData,opts.gpuid >= 0,bpfi)
  end
end

main()
