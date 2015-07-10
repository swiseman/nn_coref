#include "TH.h"
#include "luaT.h"
#include "lua.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define torch_IntTensor TH_CONCAT_STRING_3(torch.,Int,Tensor)
#define THIntTensor_(NAME)   TH_CONCAT_4(TH,Int,Tensor_,NAME)
#define cr_(NAME) TH_CONCAT_3(cr_, Real, NAME)

#include "generic/cmd_ag.c"
#include "THGenerateFloatTypes.h"

#include "generic/cr_util.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libcr(lua_State *L);

int luaopen_libcr(lua_State *L)
{
    lua_newtable(L);
    lua_pushvalue(L, -1);
    lua_setfield(L, LUA_GLOBALSINDEX, "cr");

    cr_FloatCR_init(L);
    cr_DoubleCR_init(L);
    
    cr_FloatAG_init(L);
    cr_DoubleAG_init(L);

    return 1;
}

/* compile with:  luarocks make rocks/ag-scm-1.rockspec */

