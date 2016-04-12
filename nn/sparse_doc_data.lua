require 'hdf5'

SpDMPWData = {} -- for pairwise mention data
SpDMPWData.__index = SpDMPWData

function SpDMPWData.loadFromH5(featPfx)
    spdmmd = {}
    setmetatable(spdmmd,SpDMPWData)
    local featfi = assert(hdf5.open(featPfx .. "-pw-feats.h5"))
    spdmmd.feats = featfi:read("feats"):all()
    featfi:close()
    local offsetfi = assert(hdf5.open(featPfx .. "-pw-offsets.h5"))
    spdmmd.docStarts = offsetfi:read("doc_starts"):all()
    spdmmd.mentStarts = offsetfi:read("ment_starts"):all()
    offsetfi:close()
    spdmmd.numDocs = spdmmd.docStarts:size(1)-1
    spdmmd.maxFeat = spdmmd.feats:max()
    collectgarbage()
    assert(spdmmd.feats:isContiguous())
    -- below only works if every pair actually has same number of features...
    spdmmd.numNZ = spdmmd.mentStarts[2] - spdmmd.mentStarts[1]
    return spdmmd
end


function SpDMPWData.makeFromTensors(feats,docStarts,mentStarts) -- for debugging
    spdmmd = {}
    setmetatable(spdmmd,SpDMPWData)
    spdmmd.feats = feats
    spdmmd.docStarts = docStarts
    spdmmd.mentStarts = mentStarts
    spdmmd.numDocs = spdmmd.docStarts:size(1)-1
    spdmmd.maxFeat = spdmmd.feats:max()
    spdmmd.numNZ = spdmmd.mentStarts[2] - spdmmd.mentStarts[1]
    collectgarbage()
    assert(spdmmd.feats:isContiguous())
    return spdmmd
end


-- d1m2m1, d1m3m1, d1m3m2, d1m4m1, d1m4m2, d1m4m3
-- this assumes self.docStarts and self.mentStarts begin at 0 rather than 1
function SpDMPWData:getFeats(d,m,a)
    local docStartIdx = self.docStarts[d] --idx within self.MentStarts 1 behind  where this doc starts
    local mentAntOffset = (m-2)*(m-1)/2 + a
    return self.feats:sub(self.mentStarts[docStartIdx+mentAntOffset]+1, self.mentStarts[docStartIdx+mentAntOffset+1])
end


function SpDMPWData:numMents(d) -- solve the quadratic equation (for the positive root)
    -- we want m such that m*(m-1)/2 = numPairs => m^2 -m -2*numPairs = 0
    local numPairs = self.docStarts[d+1] - self.docStarts[d]
    return (1 + math.sqrt(1 + 4*2*numPairs))/2
end

function SpDMPWData:getMentBatch(d,m)
  local docStartIdx = self.docStarts[d] --idx within self.MentStarts 1 behind  where this doc starts
  --local mentAntOffset = (m-2)*(m-1)/2 + a
  local mentOffset = (m-2)*(m-1)/2
  return self.feats:sub(self.mentStarts[docStartIdx+mentOffset+1]+1, self.mentStarts[docStartIdx+mentOffset+m]):view(m-1,self.numNZ)
end  

function SpDMPWData:getDocBatch(d)
  local docStartIdx = self.docStarts[d]
  local nextDocStartIdx = self.docStarts[d+1]
  local feats = self.feats:sub(self.mentStarts[docStartIdx+1]+1, self.mentStarts[nextDocStartIdx+1])
  local numRows = feats:size(1)/self.numNZ
  return feats:view(numRows, self.numNZ)
end

SpDMData = {} -- for just mention data
SpDMData.__index = SpDMData

function SpDMData.loadFromH5(featPfx)
    spdmd = {}
    setmetatable(spdmd,SpDMData)
    local featfi = assert(hdf5.open(featPfx .. "-na-feats.h5"))
    spdmd.feats = featfi:read("feats"):all()
    featfi:close()
    local offsetfi = assert(hdf5.open(featPfx .. "-na-offsets.h5"))
    spdmd.docStarts = offsetfi:read("doc_starts"):all()
    spdmd.mentStarts = offsetfi:read("ment_starts"):all()
    spdmd.numDocs = spdmd.docStarts:size(1)-1
    offsetfi:close()
    spdmd.maxFeat = spdmd.feats:max()
    collectgarbage()
    assert(spdmd.feats:isContiguous())
    return spdmd
end

function SpDMData.makeFromTensors(feats,docStarts,mentStarts)
    spdmd = {}
    setmetatable(spdmd,SpDMData)
    spdmd.feats = feats
    spdmd.docStarts = docStarts
    spdmd.mentStarts = mentStarts
    spdmd.numDocs = spdmd.docStarts:size(1)-1
    spdmd.maxFeat = spdmd.feats:max()
    collectgarbage()
    assert(spdmd.feats:isContiguous())
    return spdmd
end

-- d1m2, d1m3,...
-- assumes mentStarts and docStarts start at 0
function SpDMData:getFeats(d,m)
    local docStartIdx = self.docStarts[d] --idx within self.MentStarts that this doc starts
    return self.feats:sub(self.mentStarts[docStartIdx+m-1]+1, self.mentStarts[docStartIdx+m])
end


function SpDMData:numMents(d)
    return (self.docStarts[d+1] - self.docStarts[d]) + 1
end

function SpDMData:docMiniBatch(d) -- this will only work if each mention has same # of features
    local docStartIdx = self.docStarts[d]
    local numMents = self:numMents(d)
    local docFeats = self.feats:sub(self.mentStarts[docStartIdx+1]+1,self.mentStarts[docStartIdx+numMents])
    local numCols = docFeats:size(1)/(numMents-1)
    return docFeats:view(numMents-1,numCols)
end

do
  local SpKFDMData = torch.class('SpKFDMData')
  
  -- just gonna do a hacky thing for 2 constructors
  function SpKFDMData:__init(featPfx,docStarts,numNZ,feats)
    if featPfx ~= nil then
      local featfi = assert(hdf5.open(featPfx .. "-na-feats.h5"))
      local allfeats = featfi:read("feats"):all():long()
      featfi:close()
      local offsetfi = assert(hdf5.open(featPfx .. "-na-offsets.h5"))
      self.docStarts = offsetfi:read("doc_starts"):all()
      local mentStarts = offsetfi:read("ment_starts"):all()
      self.numNZ = mentStarts[2] - mentStarts[1]
      self.numDocs = self.docStarts:size(1)-1
      offsetfi:close()
      self.maxFeat = allfeats:max()
      self.feats = allfeats:view(allfeats:size(1)/self.numNZ, self.numNZ)
    else
      self.docStarts = docStarts
      self.numNZ = numNZ
      self.numDocs = self.docStarts:size(1)-1
      self.maxFeat = feats:max()
      self.feats = feats:view(feats:size(1)/self.numNZ, self.numNZ) 
    end
    collectgarbage()
    assert(self.feats:isContiguous())       
  end
  
  function SpKFDMData:getFeats(d,m)
    return self.feats[self.docStarts[d]+m]
  end
  
  function SpKFDMData:docBatch(d)
    return self.feats:sub(self.docStarts[d]+1,self.docStarts[d+1])
  end

  function SpKFDMData:numMents(d)
    return (self.docStarts[d+1] - self.docStarts[d])
  end  
  
  function SpKFDMData:cudify()
    self.feats = self.feats:cuda()
    self.docStarts = self.docStarts:cuda()
    collectgarbage()
    assert(self.feats:getDevice() ~= nil)
    assert(self.docStarts:getDevice() ~= nil)
  end

end