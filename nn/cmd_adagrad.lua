--[[
   Composite Mirror Descent (with L1 regularization) AdaGrad update.
   Several implementations of equation (25) in "Adaptive Subgradient Methods for
   Online Learning and Stochastic Optimization," Duchi et al.
   (http://www.magicbroom.info/Papers/DuchiHaSi10.pdf)
--]]


--[[
 -- a simple, somewhat memory-inefficient implementation (though no worse than optim's adagrad).
 works like all the other optim stuff
ARGS:
- `opfunc` : a function that takes a single input (X), the point of
         evaluation, and returns f(X) and df/dX
         N.B. f(X) is completely ignored, so pass in anything you want
- `x` : the initial point.
- `config.lamb` : L1 regularization coefficient
- `config.eta` : learning rate
- `state` : a table describing the state of the optimizer; after each
         call the state is modified
- `state.paramVariance` : vector of temporal variances of parameters

RETURN:
- `x` : the new x vector
- `f(x)` : the function, evaluated before the update
]]
function cmdAdaGrad(opfunc, x, config, state)
   -- (0) get/update state
   if config == nil and state == nil then
      print('no state table, cmdAdaGrad initializing')
   end
   local config = config or {}
   local state = state or config
   local eta = config.eta or 1e-3
   local lamb = config.lamb or 1e-5
   state.evalCounter = state.evalCounter or 0

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)
      
   -- (2) update per-param variance
   if not state.paramVariance then
      state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
      state.h = torch.Tensor():typeAs(x):resizeAs(dfdx):zero() -- stores 1 + stddev
   end
   
   state.paramVariance:addcmul(1,dfdx,dfdx)
   
   -- (3) let h = 1 + sqrt(variance)
   state.h:copy(state.paramVariance)
   state.h:sqrt()
   state.h:add(1)
   
   -- (4) do update: x = x - eta*dfdx/h
   x:addcdiv(-eta,dfdx,state.h)
   
   -- (5) do thresholding (for regularization)
   dfdx:sign(x) -- use dfdx to store the sign (since we're done with it anyway)
   state.h:pow(-1) -- take reciprocal of h (there's probably a better way of doing this)
   -- take magnitude and subtract off regularization term
   x:abs()
   x:add(-lamb*eta,state.h)
   -- if we're negative, set to 0
   x:clamp(0,math.huge) -- is there an elementwise maximum?
   -- finally, multiply by sign so we recover original gradient
   x:cmul(dfdx) -- recall that dfdx stores the sign now
   
   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end

--[[ all the following functions assume the input (x) and the subgradient (g) are contiguous
     in memory --]]

--[[
 -- a more memory-efficient version (and generally faster) implementation; calls a C function.
ARGS:
- `x` : the initial point.
- `g` : (sub)gradient at x
- `eta` : learning rate
- `lamb` : L1 regularization coefficient
- `state` : a table describing the state of the optimizer; after each
         call the state is modified
- `state.paramVariance` : vector of temporal variances of parameters

RETURN:
 - nothing
]]

function cmdAdaGradC(x, g, eta, lamb, state)   
   assert(x:isContiguous())
   assert(g:isContiguous())
   local state = state or {}
   local eta = eta or 1e-3
   local lamb = lamb or 1e-5

   -- update per-param variance
   if not state.paramVariance then
      state.paramVariance = torch.Tensor():typeAs(x):resizeAs(g):zero()
   end
   assert(state.paramVariance:isContiguous())
   state.paramVariance:addcmul(1,g,g)
   
   -- call c function
   x.cr.AG_cmd_adagrad_step(x,g,state.paramVariance,eta,lamb)

end


--[[
 -- sparse version of c-based adagrad implementation. expects nz indices as keys in a table.
ARGS:
- `x` : the initial point.
- `g` : (sub)gradient at x
- `nzIdxs` : a table with non-zero feature indices as KEYS. 
- `eta` : learning rate
- `lamb` : L1 regularization coefficient
- `state` : a table describing the state of the optimizer; after each
         call the state is modified
- `state.paramVariance` : vector of temporal variances of parameters

RETURN:
 - nothing
]]

function tblCmdAdaGrad(x, g, nzIdxs, eta, lamb, state)  
   assert(x:isContiguous())
   assert(g:isContiguous())  
   local state = state or {}
   local eta = eta or 1e-3
   local lamb = lamb or 1e-5

   -- update per-param variance
   if not state.paramVariance then
      state.paramVariance = torch.Tensor():typeAs(x):resizeAs(g):zero()
   end
   assert(state.paramVariance:isContiguous())
   x.cr.AG_tbl_update_var(state.paramVariance,g,nzIdxs)
   x.cr.AG_tbl_cmd_adagrad_step(x,g,state.paramVariance,eta,lamb,nzIdxs)

end


--[[
 -- sparse version of c-based adagrad implementation. assumes COLUMNS (not rows)
    are sparse, and so is suitable for use on an nn.LookupTable (rather than
    an nn.SparseLinear) layer. expects nz indices as keys in a table.
ARGS:
- `x` : the initial point.
- `g` : (sub)gradient at x
- `nzIdxs` : a table with non-zero feature indices as KEYS. 
- `eta` : learning rate
- `lamb` : L1 regularization coefficient
- `state` : a table describing the state of the optimizer; after each
         call the state is modified
- `state.paramVariance` : vector of temporal variances of parameters

RETURN:
 - nothing
]]

function colTblCmdAdaGrad(x, g, nzIdxs, eta, lamb, state)
   assert(x:isContiguous())
   assert(g:isContiguous())   
   local state = state or {}
   local eta = eta or 1e-3
   local lamb = lamb or 1e-5

   -- update per-param variance
   if not state.paramVariance then
      state.paramVariance = torch.Tensor():typeAs(x):resizeAs(g):zero()
   end
   assert(state.paramVariance:isContiguous())
   x.cr.AG_col_tbl_update_var(state.paramVariance,g,nzIdxs)
   x.cr.AG_col_tbl_cmd_adagrad_step(x,g,state.paramVariance,eta,lamb,nzIdxs)

end
