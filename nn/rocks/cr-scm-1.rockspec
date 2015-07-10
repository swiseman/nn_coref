package = "cr"
version = "scm-1"

source = {
   url = "git://github.com/swiseman/nn_coref",
}

description = {
   summary = "Coref Utils",
   detailed = [[
   ]],
   homepage = "",
   license = "See LICENSE.txt"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
