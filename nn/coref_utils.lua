--[[ Oracle Predicted Clusters... --]]

OPClust = {} -- class representing oracle predicted clustering (for a doc)
OPClust.__index = OPClust

function OPClust.makeFromStr2(s,numMents)
   opc = {}
   setmetatable(opc, OPClust)
   opc.m2c = torch.zeros(numMents)
   opc.clusts = {} -- clusters in order
   
   local clustStrs = split2(s,"|")
   for i,clustStr in ipairs(clustStrs) do
       local clustIdxs = split2(clustStr, " ")
       table.insert(opc.clusts, torch.zeros(#clustIdxs))
       for j,idx in ipairs(clustIdxs) do
           local idxNum = tonumber(idx) + 1 -- make one-indexed
           opc.m2c[idxNum] = i
           opc.clusts[i][j] = idxNum
       end 
   end  
   collectgarbage()
   return opc  
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

function getOPCs(fi,spDocData)
    local OPCs = {}
    local d = 1
    for line in io.lines(fi) do
        table.insert(OPCs, OPClust.makeFromStr2(line,spDocData:numMents(d)))
        d = d+1
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

--[[ matrix initialization --]]

-- assumes 2d and that the longer dimension should be sparsely filled in
function sparseSutsMatInit(W,numNZ,scale)
    local numNZ = numNZ or 15
    local scale = scale or 0.25
    local m = W:size(1)
    local n = W:size(2)
    -- zero everything out
    W:fill(0)
    if n >= m then -- assume columns are features and rows are hidden dims
        for i = 1, m do
            local perm = torch.randperm(n)
            -- probably better ways of doing this
            local r = torch.randn(numNZ)*scale
            for j = 1, numNZ do
                W[i][perm[j]] = r[j]
            end
        end
    else -- assume rows are features and columns hidden dims
        for j = 1, n do
            local perm = torch.randperm(m)
            local r = torch.randn(numNZ)*scale
            for i = 1, numNZ do
                W[perm[i]][j] = r[i]
            end
        end
    end
end

function checkContigAndSutsInit(net,numNZ)
    local numNZ = numNZ or 15
    -- make sure contiguous, and do sparse sutskever init while we're at it
    for layer, mod in ipairs(net.modules) do
        if mod.weight and mod.weight:size(1) > 1 then -- exclude vectors that are sometimes 1 x n
            assert(mod.weight:isContiguous())
            sparseSutsMatInit(mod.weight,numNZ,0.25)
        elseif mod.weight and mod.weight:size(1) == 1 then
            assert(mod.weight:isContiguous())
            mod.weight[1]:randn(mod.weight:size(2))
        end
        if mod.bias then
            assert(mod.bias:isContiguous())
            mod.bias:fill(0.5)
        end
    end
end

function checkContig(net)
    for layer, mod in ipairs(net.modules) do
        if mod.weight then -- exclude vectors that are sometimes 1 x n
            assert(mod.weight:isContiguous())
        end
        if mod.bias then
            assert(mod.bias:isContiguous())
        end
    end
end


--[[ misc --]]

function setFromList(tbl)
  s = {}
  for i = 1, #tbl do
    s[tbl[i]] = true
  end
  return s
end

-- adds keys from src tensor to dst table/set
function addKeys(srcT, dstSet)
    local t_data = torch.data(srcT)
    for i = 0, srcT:size(1)-1 do
        dstSet[t_data[i]] = true
    end
end

-- Compatibility: Lua-5.0
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