#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cr_util.c"
#else


#ifndef CR_H
#define CR_H

#include <stdio.h>
#include <math.h>
#include <omp.h>


// clust is a double array, but really contains indices of antecedents
// this will return a 1-indexed antecedent index
int max_gold_ant(int m, int start, double *scores, double *clust){
    int best_ant = (int)(clust[0]);
    double best_score = scores[start+best_ant-1];
    int ant_idx = 1;
    while(clust[ant_idx] < m){
        int curr_ant = (int)(clust[ant_idx]);
        if (scores[curr_ant-1] > best_score){
            best_ant = curr_ant;
            best_score = scores[start+curr_ant-1];
        }
        ant_idx += 1;
    }
    return best_ant;
}


// assume mentions are 1-indexed
inline double cost(int m, int a, double *clust, double *m2c, double fl, double fn, double wl){
   if (clust[0] == m && a != m) { // not anaphoric, so fl
       return fl;
   } else if (clust[0] < m && a == m){ // anaphoric, so fn
       return fn;
   } else if (m2c[m-1] != m2c[a-1]) { // must have predicted a link and gotten it wrong, so wl
       return wl;
   }
   return 0;
}


// assumes mentions are 1-indexed
// returns 1-indexed mention w/ greatest loss
int mult_la_argmax(int m, int late, int start, double *scores, double *clust, double *m2c, double fl, double fn, double wl){
    int best_idx = 1;
    double most_loss = cost(m,best_idx,clust,m2c,fl,fn,wl)*(1+scores[start+best_idx-1]-scores[start+late-1]);
    for (int i = 2; i <= m; ++i){
        double curr_loss = cost(m,i,clust,m2c,fl,fn,wl)*(1+scores[start+i-1]-scores[start+late-1]);
        if (curr_loss > most_loss){
            best_idx = i;
            most_loss = curr_loss;
        }
    }
    return best_idx;
}


void sparse_lt_mult(double *Z1, double *LT, int *feats, int *ment_starts, 
          int h, int doc_start, int num_ments){
    int num_pairs = (num_ments*(num_ments+1))/2;
    int i;
    #pragma omp parallel for private(i) schedule(static)
    for (i = 0; i < h; ++i){
        int col = 0;
        for (int m = 0; m < num_ments; ++m){
            for (int j = 0; j < m; ++j){         
                int offset = (m-1)*(m)/2 + j; //offset in ment_starts for this pair
                int num_feats = ment_starts[doc_start+offset+1] - ment_starts[doc_start+offset];
                int feat_offset = ment_starts[doc_start+offset];
                for (int k = 0; k < num_feats; ++k){
                    int feat = feats[feat_offset+k]-1;
                    Z1[i*num_pairs + col] += LT[feat*h+i];
                }
                col += 1;
            }
            col += 1;
        }
    }
}

// assumes biases have already been added to matrix
// puts pw scores on top of ana scores in a (hp+ha) x num_pairs matrix
void calc_fm_layer1(double *Z1, double *LTp, int *pwfeats, int *pw_starts, double *LTa,
                   int *anafeats, int *ment_starts, int hp, int ha, int pw_doc_start,
                   int ana_doc_start, int num_ments){
    int num_pairs = (num_ments*(num_ments+1))/2;
    int i;
    #pragma omp parallel for private(i) schedule(static)
    for (i = 0; i < hp; ++i){
        int col = 0;
        for (int m = 0; m < num_ments; ++m){
            for (int j = 0; j < m; ++j){          
                int offset = (m-1)*(m)/2 + j; //offset in pw_starts for this pair
                int num_feats = pw_starts[pw_doc_start+offset+1] - pw_starts[pw_doc_start+offset];
                int feat_offset = pw_starts[pw_doc_start+offset];
                for (int k = 0; k < num_feats; ++k){
                    int feat = pwfeats[feat_offset+k]-1;
                    Z1[i*num_pairs + col] += LTp[feat*hp+i];
                }
                Z1[i*num_pairs + col] = tanh(Z1[i*num_pairs + col]);
                col += 1;
            }
            col += 1;
        }
    } 
    
    #pragma omp parallel for private(i) schedule(static)
    for (i = 0; i < ha; ++i){
        for (int m = 1; m < num_ments; ++m){
            int num_feats = ment_starts[ana_doc_start+m] - ment_starts[ana_doc_start+m-1];
            int feat_offset = ment_starts[ana_doc_start+m-1];
            int ment_start = (m*(m+1))/2; // corresponds to (m,0)
            for (int k = 0; k < num_feats; ++k){
                int feat = anafeats[feat_offset+k]-1;
                Z1[(hp+i)*num_pairs + ment_start] += LTa[feat*ha+i];
            }
            Z1[(hp+i)*num_pairs + ment_start] = tanh(Z1[(hp+i)*num_pairs + ment_start]);
            // now we need to broadcast the value we just calc'd for all the other pairs
            double ment_score = Z1[(hp+i)*num_pairs + ment_start];
            for (int j = 1; j < m; ++j){
                Z1[(hp+i)*num_pairs + ment_start + j] = ment_score;
            } 
        }
    }                 
}

#endif /* CR_H */


static int cr_(CR_max_gold_ant)(lua_State *L) {
    int m = luaL_checkint(L,1);
    int start = luaL_checkint(L,2);
    THTensor *scores = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *clust = luaT_checkudata(L, 4, torch_Tensor);
    int mga = max_gold_ant(m, start, THTensor_(data)(scores), THTensor_(data)(clust));
    lua_pushnumber(L,mga);
    return 1;
}

static int cr_(CR_mult_la_argmax)(lua_State *L) {
    int m = luaL_checkint(L,1);
    int late = luaL_checkint(L,2);
    int start = luaL_checkint(L,3);
    THTensor *scores = luaT_checkudata(L, 4, torch_Tensor);
    THTensor *clust = luaT_checkudata(L, 5, torch_Tensor);
    THTensor *m2c = luaT_checkudata(L, 6, torch_Tensor);
    double fl = luaL_checknumber(L,7);
    double fn = luaL_checknumber(L,8);
    double wl = luaL_checknumber(L,9);
    int mla = mult_la_argmax(m,late,start,THTensor_(data)(scores),THTensor_(data)(clust),THTensor_(data)(m2c),fl,fn,wl);
    lua_pushnumber(L,mla);
    return 1;
}


static int cr_(CR_cost)(lua_State *L) {
    int m = luaL_checkint(L,1);
    int a = luaL_checkint(L,2);
    THTensor *clust = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *m2c = luaT_checkudata(L, 4, torch_Tensor);
    double fl = luaL_checknumber(L,5);
    double fn = luaL_checknumber(L,6);
    double wl = luaL_checknumber(L,7);
    double c = cost(m,a,THTensor_(data)(clust),THTensor_(data)(m2c),fl,fn,wl);
    lua_pushnumber(L,c);
    return 1;
}

          
static int cr_(CR_sparse_lt_mult)(lua_State *L) {
    THTensor *Z1 = luaT_checkudata(L, 1, torch_Tensor);
    THTensor *LT = luaT_checkudata(L, 2, torch_Tensor);
    THIntTensor *feats = luaT_checkudata(L, 3, torch_IntTensor);
    THIntTensor *ment_starts = luaT_checkudata(L, 4, torch_IntTensor);
    int doc_start = luaL_checkint(L,5);
    int num_ments = luaL_checkint(L,6);
    int h = THTensor_(size)(LT,1); // width of LT
    sparse_lt_mult(THTensor_(data)(Z1),THTensor_(data)(LT),THIntTensor_(data)(feats),
                   THIntTensor_(data)(ment_starts),h,doc_start,num_ments);
    return 0;
}

static int cr_(CR_fm_layer1)(lua_State *L) {
    THTensor *Z1 = luaT_checkudata(L, 1, torch_Tensor);
    THTensor *LTp = luaT_checkudata(L, 2, torch_Tensor);
    THIntTensor *pwfeats = luaT_checkudata(L, 3, torch_IntTensor);
    THIntTensor *pw_starts = luaT_checkudata(L, 4, torch_IntTensor);
    THTensor *LTa = luaT_checkudata(L, 5, torch_Tensor);    
    THIntTensor *anafeats = luaT_checkudata(L, 6, torch_IntTensor);
    THIntTensor *ment_starts = luaT_checkudata(L, 7, torch_IntTensor);    
    int pw_doc_start = luaL_checkint(L,8);
    int ana_doc_start = luaL_checkint(L,9);
    int num_ments = luaL_checkint(L,10);
    int hp = THTensor_(size)(LTp,1); // width of LTp
    int ha = THTensor_(size)(LTa,1); // width of LTa
    calc_fm_layer1(THTensor_(data)(Z1),THTensor_(data)(LTp),THIntTensor_(data)(pwfeats),
                   THIntTensor_(data)(pw_starts), THTensor_(data)(LTa),
                   THIntTensor_(data)(anafeats), THIntTensor_(data)(ment_starts),
                   hp,ha,pw_doc_start,ana_doc_start,num_ments);
    return 0;
}


static const struct luaL_Reg cr_(CR__) [] = {
    {"CR_max_gold_ant", cr_(CR_max_gold_ant)},
    {"CR_mult_la_argmax", cr_(CR_mult_la_argmax)},
    {"CR_cost", cr_(CR_cost)},
    {"CR_sparse_lt_mult", cr_(CR_sparse_lt_mult)},
    {"CR_fm_layer1", cr_(CR_fm_layer1)},
    {NULL, NULL}
};

static void cr_(CR_init)(lua_State *L)
{
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, cr_(CR__), "cr");
    lua_pop(L,1);
}


#endif
