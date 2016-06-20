local stringx = require('pl.stringx')
local file = require('pl.file')

--[[ Oracle Predicted Clusters... --]]
do
  local OPClust = torch.class('OPClust')

  function OPClust:__init(s,numMents)
     self.m2c = torch.zeros(numMents)
     self.clusts = {} -- clusters in order
     self.mci = torch.zeros(numMents) -- mention cluster idx (which idx w/in its cluster a mention is)
     
     local clustStrs = split2(s,"|")
     for i,clustStr in ipairs(clustStrs) do
         local clustIdxs = split2(clustStr, " ")
         table.insert(self.clusts, torch.LongTensor(#clustIdxs):contiguous())
         for j,idx in ipairs(clustIdxs) do
             local idxNum = tonumber(idx) + 1 -- make one-indexed
             self.m2c[idxNum] = i
             self.clusts[i][j] = idxNum
             self.mci[idxNum] = j
         end 
     end  
     collectgarbage()
  end
  
  function OPClust:cudify() -- just cudifies actual clusts
    for i = 1, #self.clusts do
      self.clusts[i] = self.clusts[i]:cuda()
    end
  end

  function OPClust:anaphoric(ment) -- is anaphoric?
      return self.clusts[self.m2c[ment]][1] < ment
  end

   -- returns nil if ment is the first in the cluster (i.e., has no antecedents)
  function OPClust:ants(ment)
      local i = 0
      local clust = self.clusts[self.m2c[ment]]
      return function()
          i = i + 1        
          if clust[i] and clust[i] < ment then
              return clust[i]
          end
      end
  end

  --  fl is false link loss; fn is false new loss; wl is wrong link loss
  function OPClust:cost(ment,ant,fl,fn,wl)
      if not self:anaphoric(ment) and ment ~= ant then
          return fl -- false link
      elseif self:anaphoric(ment) and ment == ant then
          return fn -- false new
      elseif self.m2c[ment] ~= self.m2c[ant] then -- if a mistake must be wl since already checked fl
          return wl
      else
          return 0
      end 
  end

end

function getOPCs(fi,spDocData)
  local OPCs = {}
  --local d = 1
	local lines = file.read(fi)
	lines = stringx.splitlines(lines)    
  for d = 1, #lines do
    table.insert(OPCs, OPClust(lines[d],spDocData:numMents(d)))
  end
  collectgarbage()
  return OPCs
end


function maxGoldAnt(clust,scoreBuf,ment,start)
    local bestIdx = -1
    local bestScore = -math.huge
    for ant in clust:ants(ment) do
        if scoreBuf[start+ant] > bestScore then
            bestIdx = ant
            bestScore = scoreBuf[start+ant]
        end
    end
    return bestIdx    
end

-- note this returns the true antecedent idx (not a shifted one...)
function maxGoldAnt2(clust,scoreBuf,ment,start,numAnts)
    local bestIdx = ment -- by starting this way, if no antecedents w/ in legal distance, just say NA
    local bestScore = -math.huge
    local legalAntStart = math.max(1,ment-numAnts)
    
    for ant in clust:ants(ment) do
      if ant >= legalAntStart and scoreBuf[start+ant-legalAntStart+1] > bestScore then
        bestIdx = ant
        bestScore = scoreBuf[start+ant-legalAntStart+1]
      end
    end
    return bestIdx    
end

function getMaxes(mentData,mentDevData,opcs,devOPCs)
  local maxMents = 0
  local maxNumClusts = 0
  for d = 1, mentData.numDocs do
    if mentData:numMents(d) > maxMents then
      maxMents = mentData:numMents(d)
    end
    if #opcs[d].clusts > maxNumClusts then
      maxNumClusts = #opcs[d].clusts
    end
  end
  
  for d = 1, mentDevData.numDocs do
    if mentDevData:numMents(d) > maxMents then
      maxMents = mentDevData:numMents(d)
    end
    if #devOPCs[d].clusts > maxNumClusts then
      maxNumClusts = #devOPCs[d].clusts
    end   
  end
  return maxMents, maxNumClusts
end

function simpleAnteLAArgmax(m2c,scores,ment,late,start)
  local bestIdx = 1
  local mostLoss = (1 + scores[start+1] - scores[start+late])
  if m2c[1] == m2c[late] then
    mostLoss = 0
  end
  for i = 2, ment-1 do
    local antLoss = (1 + scores[start+i] - scores[start+late])
    if m2c[i] == m2c[late] then
      antLoss = 0
    end
    if antLoss > mostLoss then
      bestIdx = i
      mostLoss = antLoss
    end 
  end
  return bestIdx
end


function simpleMultLAArgmax(opc,scores,ment,lateScore,naScore,start,fl,fn,wl)
  local bestIdx
  local mostLoss
  local delt -- the cost associated with the biggest mistake we made
  if opc:anaphoric(ment) then -- only fn and wl possible
    bestIdx = ment
    mostLoss = fn*(1 + naScore - lateScore)
    delt = fn
    for a = 1, ment-1 do
      -- zero loss if in same cluster (note that 0 is not falsey in lua)
      local antLoss = (opc.m2c[ment] == opc.m2c[a]) and 0 or wl*(1 + scores[start+a] - lateScore)
      if antLoss > mostLoss then
        bestIdx = a
        mostLoss = antLoss
        delt = (opc.m2c[ment] == opc.m2c[a]) and 0 or wl
      end
    end
  else -- not anaphoric, so only fl possible
    bestIdx = ment
    mostLoss = 0
    delt = 0
    for a = 1, ment-1 do
      local antLoss = fl*(1 + scores[start+a] - lateScore)
      if antLoss > mostLoss then
        bestIdx = a
        mostLoss = antLoss
        delt = fl
      end
    end
  end
  return bestIdx, delt
end


do
  local PredClustData = torch.class('PredClustData')
  
  function PredClustData:__init(startDoc,numDocs,maxFeat,maxMents,anaKFData)
    self.numDocs = numDocs
    self.maxFeat = maxFeat
    self.maxMents = maxMents
    self.clustFeats = {}
    self.nextClust = torch.IntTensor(numDocs):fill(2)
    self.predM2C = torch.IntTensor(numDocs,maxMents)
    self.clustLens = torch.IntTensor(numDocs,maxMents) -- no more than maxMents clusters anyway
    self.clustFirstMents = torch.IntTensor(numDocs,maxMents) -- no more than maxMents clusters anyway
    self.buckets = 11
    
    for d = 1, self.numDocs do
       -- use 3 x anaFeats + 11 buckets each for size of cluster and distance to 1st thing in cluster
      local fmFeats = torch.zeros(maxFeat*3+self.buckets*2)
      fmFeats:sub(1,maxFeat):indexFill(1,anaKFData:getFeats(startDoc+d-1,1),1)
      fmFeats[3*maxFeat+1] = 1 -- do cluster size (i guess ignore distance because we don't have 0...)
      table.insert(self.clustFeats,{fmFeats})
      self.predM2C[d][1] = 1
      self.clustLens[d][1] = 1
      self.clustFirstMents[d][1] = 1
    end
  end
  
  -- update with most recent prediction
  function PredClustData:addPred(startDoc,d,m,pred,anaKFData)
    local maxFeat = self.maxFeat
    local didx = d-startDoc+1
    if pred == m then -- predicted not anaphoric
      local nextCid = self.nextClust[didx]
      self.predM2C[didx][m] = nextCid
      -- need to add features for this guy
      local fmFeats = torch.zeros(maxFeat*3+self.buckets*2)    
      fmFeats:sub(1,maxFeat):indexFill(1,anaKFData:getFeats(d,m),1)
      fmFeats[3*maxFeat+1] = 1 -- do distance feature
      table.insert(self.clustFeats[didx], fmFeats)
      self.clustLens[didx][nextCid] = 1
      self.clustFirstMents[didx][nextCid] = m
      self.nextClust[didx] = nextCid + 1   
    else -- predicted m is in pred's cluster
      local cid = self.predM2C[didx][pred]
      self.predM2C[didx][m] = cid
      -- need to update features for this cluster now
      local feats = self.clustFeats[didx][cid]
      -- want to OR third set of features with middle set of features
      feats:sub(maxFeat+1,2*maxFeat):add(feats:sub(2*maxFeat+1,3*maxFeat))
      feats:sub(maxFeat+1,2*maxFeat):clamp(0,1)
      self.clustLens[didx][cid] = self.clustLens[didx][cid] + 1
      -- finally zero out the rest
      --feats:sub(2*maxFeat+1,feats:size(1)):fill(0)
    end
  end
  
  -- returns features if we were to predict m as next thing in pred's cluster
  function PredClustData:getPredFeats(startDoc,d,m,pred,anaKFData)
    local maxFeat = self.maxFeat
    local didx = d-startDoc+1
    local cid = self.predM2C[didx][pred]
    local feats = self.clustFeats[didx][cid]
    -- zero out stuff that's changing
    feats:sub(2*maxFeat+1,feats:size(1)):fill(0)
    -- add in features for m
    feats:sub(2*maxFeat+1,3*maxFeat):indexFill(1,anaKFData:getFeats(d,m),1)
    -- do size
    feats[3*maxFeat+math.min(self.clustLens[didx][cid]+1,self.buckets)] = 1
    -- do distance from first ment
    feats[3*maxFeat+self.buckets+math.min(m-self.clustFirstMents[didx][cid],self.buckets)] = 1
    return feats
  end
  
end


function calcMaxMents(pwData,anaData,batchSize)
  local maxMentsTbl = {}
  local start = 1
  for d = 1, pwData.numDocs, batchSize do
    local endDoc = math.min(pwData.numDocs, d+batchSize-1)         
    -- figure out most mentions any of these docs have
    local maxMents = anaData:numMents(d)
    for j = d+1, endDoc do
      local tempNumMents = anaData:numMents(j)
      if tempNumMents > maxMents then
        maxMents = tempNumMents
      end
    end
    table.insert(maxMentsTbl,maxMents)
  end
  return maxMentsTbl
end

function getDocPosns(numMents,cuda)
  local docDists = torch.range(1,numMents)
  docDists:mul(2)
  docDists:add(-numMents-1)
  docDists:div(numMents-1) 
  return cuda and docDists:cuda() or docDists
end


function split2(str, delim, maxNb)
    -- Eliminate bad cases...
   if string.find(str, delim) == nil then
      return { str }
    end
    if maxNb == nil or maxNb < 1 then
        maxNb = 0    -- No limit
    end
    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local nb = 0
    local lastPos
    for part, pos in string.gfind(str, pat) do
        nb = nb + 1
        result[nb] = part
        lastPos = pos
        if nb == maxNb then break end
    end
    -- Handle the last field
    if nb ~= maxNb then
       result[nb + 1] = string.sub(str, lastPos)
    end
    return result
end
