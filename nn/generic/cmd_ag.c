#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cmd_ag.c"
#else


#ifndef AG_H
#define AG_H

#include <stdio.h>
#include <math.h>
#include <omp.h>
#define MAX_NZ 4096 /* we expect at most this many sparse features */
#define MAX(a,b) (((a)>(b))?(a):(b))

/*---------------- Implementation Notes -----------------------
 1) All these functions assume everything is contiguous 
 2) The weight corresponding to feature index i is expected to be at index i-1
 3) If using a table-based (sparse) function to update a multirow tensor
    AND the table contains more than MAX_NZ keys/indices, this code will fail
    catastrophically.
---------------------------------------------------------------*/


void cmd_adagrad_step(int n, double *x, double *g, double *var, double eta, double lamb){
     for (int i = 0; i < n; ++i){
         if (g[i] != 0) {
             double eta_ov_hti = eta/(1.0+sqrt(var[i]));
             double xi_upd = x[i] - eta_ov_hti*g[i];
             x[i] = MAX(0, (fabs(xi_upd) - lamb*eta_ov_hti));
             // negate if gradient was actually negative
             if (x[i] != 0 && xi_upd < 0) {
                 x[i] *= - 1;
             } 
         }
     }
}


void sp_update_var(double *var, double *g, int *nz, int h, int d, int num_nz){
    int i;
    #pragma omp parallel for private(i) schedule(static)
    for (i = 0; i < h; ++i) {
        for (int n = 0; n < num_nz; ++n){
            int feat = nz[n]-1;
            var[i*d+feat] += g[i*d+feat]*g[i*d+feat];
        }
   }
}


void sp_cmd_adagrad_step(double *x, double *g, double *var, int *nz, double eta, double lamb,
                                      int h, int d, int num_nz){
    int i;
    #pragma omp parallel for private(i) schedule(static)
    for (i = 0; i < h; ++i) {
        for (int n = 0; n < num_nz; ++n){
            int feat = nz[n]-1;
            if (g[i*d+feat] != 0) {
                double eta_ov_hti = eta/(1.0+sqrt(var[i*d+feat]));
                double xi_upd = x[i*d+feat] - eta_ov_hti*g[i*d+feat];
                x[i*d+feat] = MAX(0, (fabs(xi_upd) - lamb*eta_ov_hti));
                // negate if gradient was actually negative
                if (x[i*d+feat] != 0 && xi_upd < 0) {
                    x[i*d+feat] *= - 1;
                } 
            }
        }
    }
}


/* the following deal with sparse column rather than row updates; this happens when using
 a lookup table, which views its weights as the usual sparse matrix setup, but transposed.
 accordingly, for the following two functions, d is actually the height of the matrix, and
 h is the width... */

void sp_col_update_var(double *var, double *g, int *nz, int d, int h, int num_nz){
    int n;
    #pragma omp parallel for private(n) schedule(static)
    for (n = 0; n < num_nz; ++n){
        int feat = nz[n]-1; // now a row!
        for (int i = 0; i < h; ++i) {    
            var[feat*h+i] += g[feat*h+i]*g[feat*h+i];
        }
   }
}


void sp_col_cmd_adagrad_step(double *x, double *g, double *var, int *nz, double eta, double lamb,
                                      int d, int h, int num_nz){
    int n;
    #pragma omp parallel for private(n) schedule(static)
    for (n = 0; n < num_nz; ++n){
        int feat = nz[n]-1; // now a row!
        for (int i = 0; i < h; ++i) {    
            if (g[feat*h+i] != 0) {
                double eta_ov_hti = eta/(1.0+sqrt(var[feat*h+i]));
                double xi_upd = x[feat*h+i] - eta_ov_hti*g[feat*h+i];
                x[feat*h+i] = MAX(0, (fabs(xi_upd) - lamb*eta_ov_hti));
                // negate if gradient was actually negative
                if (x[feat*h+i] != 0 && xi_upd < 0) {
                    x[feat*h+i] *= - 1;
                } 
            }
        }
    }
}


/* these are just convenience functions so we don't have to change double tensors
   containing the nz idxs to ints */

void sp_update_var_dnz(double *var, double *g, double *nz, int h, int d, int num_nz){
    int i;
    #pragma omp parallel for private(i) schedule(static)
    for (i = 0; i < h; ++i) {
        for (int n = 0; n < num_nz; ++n){
            int feat = (int) (nz[n]-1);
            var[i*d+feat] += g[i*d+feat]*g[i*d+feat];
        }
   }
}


void sp_cmd_adagrad_step_dnz(double *x, double *g, double *var, double *nz, double eta, double lamb,
                                      int h, int d, int num_nz){
    int i;
    #pragma omp parallel for private(i) schedule(static)
    for (i = 0; i < h; ++i) {
        for (int n = 0; n < num_nz; ++n){
            int feat = (int) (nz[n]-1);
            if (g[i*d+feat] != 0) {
                double eta_ov_hti = eta/(1.0+sqrt(var[i*d+feat]));
                double xi_upd = x[i*d+feat] - eta_ov_hti*g[i*d+feat];
                x[i*d+feat] = MAX(0, (fabs(xi_upd) - lamb*eta_ov_hti));
                // negate if gradient was actually negative
                if (x[i*d+feat] != 0 && xi_upd < 0) {
                    x[i*d+feat] *= - 1;
                } 
            }
        }
    }
}


/* ----------------------------------------------------------------------------
 - the following functions will actually try to use lua table keys as indices -                           -
 -----------------------------------------------------------------------------*/
void tbl_update_var(double *var, double *g, int h, int d, lua_State *L, int tbl_idx){
    for (int i = 0; i < h; ++i) {
        lua_pushnil(L); // push nil key (so we can pop something with lua_next)
        while (lua_next(L,tbl_idx) != 0){
            // the above pushes the key to index -2 and val to index -1; we only care about the key
            // the following should convert to a signed integral type
            int feat = lua_tointeger(L,-2) - 1; // subtract 1 b/c now 0-indexing.
            var[i*d+feat] += g[i*d+feat]*g[i*d+feat];
            lua_pop(L,1); // removes val; key still on top
        }
        // don't need to pop again, b/c lua_next pops top thing when it returns 0
    }
}

void tbl_cmd_adagrad_step(double *x, double *g, double *var, double eta, double lamb,
                                      int h, int d, lua_State *L, int tbl_idx){
    for (int i = 0; i < h; ++i) {
        lua_pushnil(L); // push nil key (so we can pop something with lua_next)
        while (lua_next(L,tbl_idx) != 0){
            // the above pushes the key to index -2 and val to index -1; we only care about the key
            // the following should convert to a signed integral type
            int feat = lua_tointeger(L,-2) - 1; // subtract 1 b/c now 0-indexing.
            if (g[i*d+feat] != 0) {
                double eta_ov_hti = eta/(1.0+sqrt(var[i*d+feat]));
                double xi_upd = x[i*d+feat] - eta_ov_hti*g[i*d+feat];
                x[i*d+feat] = MAX(0, (fabs(xi_upd) - lamb*eta_ov_hti));
                // negate if gradient was actually negative
                if (x[i*d+feat] != 0 && xi_upd < 0) {
                    x[i*d+feat] *= - 1;
                } 
            }
            
            lua_pop(L,1); // removes val; key still on top
        }
        // don't need to pop again, b/c lua_next pops top thing when it returns 0
    }
}


#endif /* AG_H */


/* we assume we're dealing with at most 2d tensors */

static int cr_(AG_cmd_adagrad_step)(lua_State *L) {
    THTensor *x = luaT_checkudata(L, 1, torch_Tensor);
    THTensor *g = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *var = luaT_checkudata(L, 3, torch_Tensor);
    double eta = luaL_checknumber(L,4);
    double lamb = luaL_checknumber(L,5);

    int n = THTensor_(nElement)(x);

    cmd_adagrad_step(n, THTensor_(data)(x), THTensor_(data)(g), THTensor_(data)(var),
               eta, lamb);
    return 0;
}


static int cr_(AG_sp_update_var)(lua_State *L) {
    THTensor *var = luaT_checkudata(L, 1, torch_Tensor);
    THTensor *g = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *nz = luaT_checkudata(L, 3, torch_Tensor);

    int h,d;
    if (THTensor_(nDimension)(var) > 1) {
        h = THTensor_(size)(var,0); // height
        d = THTensor_(size)(var,1); // width
    } else {
        h = 1;
        d = THTensor_(nElement)(var);
    }
    int num_nz = THTensor_(nElement)(nz);
    
    sp_update_var_dnz(THTensor_(data)(var), THTensor_(data)(g), THTensor_(data)(nz), h, d, num_nz);

    return 0;
}


static int cr_(AG_sp_cmd_adagrad_step)(lua_State *L) {
    THTensor *x = luaT_checkudata(L, 1, torch_Tensor);
    THTensor *g = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *var = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *nz = luaT_checkudata(L, 4, torch_Tensor);
    double eta = luaL_checknumber(L,5);
    double lamb = luaL_checknumber(L,6);

    int h,d;
    if (THTensor_(nDimension)(x) > 1) {
        h = THTensor_(size)(x,0); // height
        d = THTensor_(size)(x,1); // width
    } else {
        h = 1;
        d = THTensor_(nElement)(x);
    }
    int num_nz = THTensor_(nElement)(nz);
    
    sp_cmd_adagrad_step_dnz(THTensor_(data)(x),THTensor_(data)(g),THTensor_(data)(var),THTensor_(data)(nz),eta,lamb,h,d,num_nz);    

    return 0;
}

/* global array so we don't have to reallocate all the time; used by table-based functions */
int global_nz[MAX_NZ];


/* expects a lua table as the third function argument */
static int cr_(AG_tbl_update_var)(lua_State *L) {
    THTensor *var = luaT_checkudata(L, 1, torch_Tensor);
    THTensor *g = luaT_checkudata(L, 2, torch_Tensor);

    int h, d, feat_idx;
    if (THTensor_(nDimension)(var) > 1) {
        h = THTensor_(size)(var,0); // height
        d = THTensor_(size)(var,1); // width
    } else {
        h = 1;
        d = THTensor_(nElement)(var);
    }
    
    // if just one row, might as well use the table; otherwise we want to parallelize,
    // so we copy to an array that can easily be shared between threads
    if (h == 1) {
       tbl_update_var(THTensor_(data)(var), THTensor_(data)(var), h, d, L, 3);
    } else { // copy the feature idxs from the lua table
        feat_idx = 0;
        lua_pushnil(L); // push nil key (so we can pop something with lua_next)
        while (lua_next(L,3) != 0){
            // the above pushes the key to index -2 and val to index -1; we only care about the key
            // the following should convert to a signed integral type
            int feat = lua_tointeger(L,-2); // leave 1-based indexing for now.
            if (feat_idx >= MAX_NZ){
               printf("ERROR: expecting at most %d sparse features; about to seg-fault", MAX_NZ);
            }
            global_nz[feat_idx] = feat;
            lua_pop(L,1); // removes val; key still on top
            feat_idx += 1;
        } 
        sp_update_var(THTensor_(data)(var), THTensor_(data)(g), global_nz, h, d, feat_idx);         
    }

    return 0;
}

/* expects a lua table as the 6th function argument */
static int cr_(AG_tbl_cmd_adagrad_step)(lua_State *L) {
    THTensor *x = luaT_checkudata(L, 1, torch_Tensor);
    THTensor *g = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *var = luaT_checkudata(L, 3, torch_Tensor);
    double eta = luaL_checknumber(L,4);
    double lamb = luaL_checknumber(L,5);

    int h, d, feat_idx;
    if (THTensor_(nDimension)(x) > 1) {
        h = THTensor_(size)(x,0); // height
        d = THTensor_(size)(x,1); // width
    } else {
        h = 1;
        d = THTensor_(nElement)(x);
    }
    
    // if just one row, might as well use the table; otherwise we want to parallelize,
    // so we copy to an array that can easily be shared between threads
    if (h == 1) {
       tbl_cmd_adagrad_step(THTensor_(data)(x),THTensor_(data)(g),THTensor_(data)(var),eta,lamb,h,d,L,6);
    } else { // copy the feature idxs from the lua table
        feat_idx = 0;
        lua_pushnil(L); // push nil key (so we can pop something with lua_next)
        while (lua_next(L,6) != 0){
            // the above pushes the key to index -2 and val to index -1; we only care about the key
            // the following should convert to a signed integral type
            int feat = lua_tointeger(L,-2); // leave 1-based indexing for now.
            if (feat_idx >= MAX_NZ){
               printf("ERROR: expecting at most %d sparse features; about to seg-fault", MAX_NZ);
            }            
            global_nz[feat_idx] = feat;
            lua_pop(L,1); // removes val; key still on top
            feat_idx += 1;
        }       
        sp_cmd_adagrad_step(THTensor_(data)(x),THTensor_(data)(g),THTensor_(data)(var),global_nz,eta,lamb,h,d,feat_idx);         
    }        

    return 0;
}



/* expects a lua table as the third function argument */
static int cr_(AG_col_tbl_update_var)(lua_State *L) {
    THTensor *var = luaT_checkudata(L, 1, torch_Tensor);
    THTensor *g = luaT_checkudata(L, 2, torch_Tensor);

    int h, d, feat_idx;
    if (THTensor_(nDimension)(var) > 1) {
        d = THTensor_(size)(var,0); // height
        h = THTensor_(size)(var,1); // width
    } else {
        h = 1;
        d = THTensor_(nElement)(var);
    }
    
     // copy the feature idxs from the lua table
     feat_idx = 0;
     lua_pushnil(L); // push nil key (so we can pop something with lua_next)
     while (lua_next(L,3) != 0){
        // the above pushes the key to index -2 and val to index -1; we only care about the key
        // the following should convert to a signed integral type; maybe not int tho?
        int feat = lua_tointeger(L,-2); // leave 1-based indexing for now.
        if (feat_idx >= MAX_NZ){
          printf("ERROR: expecting at most %d sparse features; about to seg-fault", MAX_NZ);
        }           
        global_nz[feat_idx] = feat;
        lua_pop(L,1); // removes val; key still on top
        feat_idx += 1;
     } 
     sp_col_update_var(THTensor_(data)(var), THTensor_(data)(g), global_nz, d, h, feat_idx);         
    
    return 0;
}

/* expects a lua table as the 6th function argument */
static int cr_(AG_col_tbl_cmd_adagrad_step)(lua_State *L) {
    THTensor *x = luaT_checkudata(L, 1, torch_Tensor);
    THTensor *g = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *var = luaT_checkudata(L, 3, torch_Tensor);
    double eta = luaL_checknumber(L,4);
    double lamb = luaL_checknumber(L,5);

    int h, d, feat_idx;
    if (THTensor_(nDimension)(x) > 1) {
        d = THTensor_(size)(x,0); // height
        h = THTensor_(size)(x,1); // width
    } else {
        h = 1;
        d = THTensor_(nElement)(var);
    }
    
    // copy the feature idxs from the lua table
    feat_idx = 0;
    lua_pushnil(L); // push nil key (so we can pop something with lua_next)
    while (lua_next(L,6) != 0){
        // the above pushes the key to index -2 and val to index -1; we only care about the key
        // the following should convert to a signed integral type
        int feat = lua_tointeger(L,-2); // leave 1-based indexing for now.
        if (feat_idx >= MAX_NZ){
            printf("ERROR: expecting at most %d sparse features; about to seg-fault", MAX_NZ);
        }           
        global_nz[feat_idx] = feat;
        lua_pop(L,1); // removes val; key still on top
        feat_idx += 1;
    }       
    sp_col_cmd_adagrad_step(THTensor_(data)(x),THTensor_(data)(g),THTensor_(data)(var),global_nz,eta,lamb,d,h,feat_idx);         
           

    return 0;
}


static const struct luaL_Reg cr_(AG__) [] = {
    {"AG_cmd_adagrad_step", cr_(AG_cmd_adagrad_step)},
    {"AG_sp_update_var", cr_(AG_sp_update_var)},
    {"AG_sp_cmd_adagrad_step", cr_(AG_sp_cmd_adagrad_step)},
    {"AG_tbl_update_var", cr_(AG_tbl_update_var)}, 
    {"AG_tbl_cmd_adagrad_step", cr_(AG_tbl_cmd_adagrad_step)},
    {"AG_col_tbl_update_var", cr_(AG_col_tbl_update_var)}, 
    {"AG_col_tbl_cmd_adagrad_step", cr_(AG_col_tbl_cmd_adagrad_step)},  
    {NULL, NULL}
};

static void cr_(AG_init)(lua_State *L)
{
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, cr_(AG__), "cr");
    lua_pop(L,1);
}


#endif
