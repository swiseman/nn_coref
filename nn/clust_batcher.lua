require 'nn'
require 'sparse_doc_data'
require 'coref_utils'

do
  local ClustAndDistBatcher = torch.class('ClustAndDistBatcher')
  
  function ClustAndDistBatcher:__init(OPCs,mentData)
    -- want to batch together all clusters of a certain size
    self.docIdxs = {}
    self.docPosns = {}
    self.docLocs = {}
    local joiner = nn.JoinTable(1,1)
    local joiner2 = nn.JoinTable(1,2)
    for d = 1, mentData.numDocs do
      local numMents = mentData:numMents(d)
      local clustIdxs = {}
      local clustDists = {} -- distances of each mention in the cluster
      local clustLocations = torch.LongTensor(#OPCs[d].clusts,2)
      for i, clust in ipairs(OPCs[d].clusts) do
        local size = clust:size(1)
        clustLocations[i][1] = size
        if clustIdxs[size] == nil then
          clustIdxs[size] = {}
          clustDists[size] = {}
        end
        table.insert(clustIdxs[size],clust)
        -- will calc dist as (2k-n-1)/(n-1), where k is numberth mention and n is numMents
        local dist = torch.Tensor.new()-- make it double/float
        dist:resize(clust:size(1),1):copy(clust)
        dist:mul(2)
        dist:csub(numMents+1)
        dist:div(numMents-1)
        table.insert(clustDists[size],dist)
        clustLocations[i][2] = #clustIdxs[size] -- numberth cluster of this particular size
      end
      -- turn clustIdxs into a tensor (for each size)
      for size, tbl in pairs(clustIdxs) do
        local numClusts = #tbl
        clustIdxs[size] = joiner:forward(tbl):reshape(numClusts,size):long()
        clustDists[size] = joiner2:forward(clustDists[size]):clone()
      end
      collectgarbage()
      table.insert(self.docIdxs,clustIdxs)
      table.insert(self.docPosns,clustDists)
      table.insert(self.docLocs,clustLocations)
    end
  end
  
  function ClustAndDistBatcher:cudify()
    for d,tbl in ipairs(self.docIdxs) do
      for size, T in pairs(tbl) do
        self.docIdxs[d][size] = T:cuda()
        -- docPosns are stored in the same way..
        self.docPosns[d][size] = self.docPosns[d][size]:cuda()
      end
    end
  end    
  
  function ClustAndDistBatcher.makePredClust(preds,numMents)
    -- DON'T FORGET TO HAVE PREDICTED FIRST MENTION
    assert(preds:size(1) == numMents)
    local docClusts = {}
    local cids = torch.ones(numMents)
    local currCid = 1
    for m = 1, numMents do
      if preds[m] == m then -- NA, so starts new cluster
        cids[m] = currCid
        docClusts[currCid] = {m}
        currCid = currCid + 1
      else
        local predCid = cids[preds[m]]
        cids[m] = predCid
        table.insert(docClusts[predCid],m)
      end
    end
    local predClust = {}
    predClust.clusts = docClusts
    for c, clust in ipairs(predClust.clusts) do
      predClust.clusts[c] = torch.LongTensor(clust)
    end
    predClust.m2c = cids
    return predClust
  end  
  
end


do
  local DocClustAndDistBatch = torch.class('DocClustAndDistBatch')
  
  function DocClustAndDistBatch:__init(d,PC,numMents)
    -- want to batch together all clusters of a certain size
    self.docIdxs = {}
    self.docPosns = {}
    self.docLocs = {}
    local joiner = nn.JoinTable(1,1)
    local joiner2 = nn.JoinTable(1,2)
    local clustIdxs = {}
    local clustDists = {} -- distances of each mention in the cluster
    local clustLocations = torch.LongTensor(#PC.clusts,2)
    for i, clust in ipairs(PC.clusts) do
      local size = clust:size(1)
      clustLocations[i][1] = size
      if clustIdxs[size] == nil then
        clustIdxs[size] = {}
        clustDists[size] = {}
      end
      table.insert(clustIdxs[size],clust)
      -- will calc dist as (2k-n-1)/(n-1), where k is numberth mention and n is numMents
      local dist = torch.Tensor.new()-- make it double/float
      dist:resize(clust:size(1),1):copy(clust)
      dist:mul(2)
      dist:csub(numMents+1)
      dist:div(numMents-1)
      table.insert(clustDists[size],dist)
      clustLocations[i][2] = #clustIdxs[size] -- numberth cluster of this particular size
    end
    -- turn clustIdxs into a tensor (for each size)
    for size, tbl in pairs(clustIdxs) do
      local numClusts = #tbl
      clustIdxs[size] = joiner:forward(tbl):reshape(numClusts,size):long()
      clustDists[size] = joiner2:forward(clustDists[size]):clone()
    end
    collectgarbage()
    self.docIdxs[d] = clustIdxs
    self.docPosns[d] = clustDists
    self.docLocs[d] = clustLocations
  end
  
  function DocClustAndDistBatch:cudify()
    for d,tbl in pairs(self.docIdxs) do
      for size, T in pairs(tbl) do
        self.docIdxs[d][size] = T:cuda()
        -- docPosns are stored in the same way..
        self.docPosns[d][size] = self.docPosns[d][size]:cuda()
      end
    end
  end    
  
  function DocClustAndDistBatch.makePredClust(preds,numMents)
    -- DON'T FORGET TO HAVE PREDICTED FIRST MENTION
    assert(preds:size(1) == numMents)
    local docClusts = {}
    local cids = torch.ones(numMents)
    local currCid = 1
    for m = 1, numMents do
      if preds[m] == m then -- NA, so starts new cluster
        cids[m] = currCid
        docClusts[currCid] = {m}
        currCid = currCid + 1
      else
        local predCid = cids[preds[m]]
        cids[m] = predCid
        table.insert(docClusts[predCid],m)
      end
    end
    local predClust = {}
    predClust.clusts = docClusts
    for c, clust in ipairs(predClust.clusts) do
      predClust.clusts[c] = torch.LongTensor(clust)
    end
    predClust.m2c = cids
    return predClust
  end  
  
end



do
  local ClustAndMoarBatcher = torch.class('ClustAndMoarBatcher')
  
  function ClustAndMoarBatcher:__init(OPCs,mentData,avgMentEmbeddings)
    -- want to batch together all clusters of a certain size
    self.docIdxs = {}
    self.docMoars = {}
    self.docLocs = {}
    local joiner = nn.JoinTable(1,1)
    local joiner2 = nn.JoinTable(1,2)
    for d = 1, mentData.numDocs do
      local numMents = mentData:numMents(d)
      local clustIdxs = {}
      local clustMoars = {} -- distances of each mention in the cluster
      local clustLocations = torch.LongTensor(#OPCs[d].clusts,2)
      for i, clust in ipairs(OPCs[d].clusts) do
        local size = clust:size(1)
        clustLocations[i][1] = size
        if clustIdxs[size] == nil then
          clustIdxs[size] = {}
          clustMoars[size] = {}
        end
        table.insert(clustIdxs[size],clust)
        clustLocations[i][2] = #clustIdxs[size] -- numberth cluster of this particular size        
        -- our additional features will include normalized numberth mention and avg embedding
        local moar = torch.Tensor.new():resize(clust:size(1),1+avgMentEmbeddings[1]:size(2))
        -- will calc  position as (2k-n-1)/(n-1), where k is numberth mention and n is numMents        
        moar:select(2,1):copy(clust)
        moar:select(2,1):mul(2)
        moar:select(2,1):add(-numMents -1)
        moar:select(2,1):div(numMents-1)
        moar[{{},{2,moar:size(2)}}] = avgMentEmbeddings[d]:index(1,clust)
        table.insert(clustMoars[size],moar)

      end
      -- turn clustIdxs into a tensor (for each size)
      for size, tbl in pairs(clustIdxs) do
        local numClusts = #tbl
        clustIdxs[size] = joiner:forward(tbl):reshape(numClusts,size):long()
        clustMoars[size] = joiner2:forward(clustMoars[size]):clone()
      end
      collectgarbage()
      table.insert(self.docIdxs,clustIdxs)
      table.insert(self.docMoars,clustMoars)
      table.insert(self.docLocs,clustLocations)
    end
  end
  
  function ClustAndMoarBatcher:cudify()
    for d,tbl in ipairs(self.docIdxs) do
      for size, T in pairs(tbl) do
        self.docIdxs[d][size] = T:cuda()
        -- docPosns are stored in the same way..
        self.docMoars[d][size] = self.docMoars[d][size]:cuda()
      end
    end
  end    
  
end
