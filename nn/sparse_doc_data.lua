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
