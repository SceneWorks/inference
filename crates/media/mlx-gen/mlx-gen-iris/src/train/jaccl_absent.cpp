// The JACCL entry points MLX's distributed backend switch references, for an MLX build that ships
// without libjaccl.a (the prebuilt cells): JACCL is unavailable and initializes no group, exactly
// as MLX's own `no_jaccl.cpp` reports when JACCL is compiled out. See build.rs (sc-25689).
//
// Signatures: exactly the pinned MLX's (v0.32.0, fetched by the pinned mlx-sys) declarations in
// `mlx/distributed/jaccl/lib/jaccl/jaccl.h` — `namespace jaccl { class Group; bool is_available();
// std::shared_ptr<Group> init(bool strict = false); }` — the two symbols `mlx/distributed/jaccl/
// jaccl.cpp` (compiled into libmlx.a when MACOSX_DEPLOYMENT_TARGET >= 26.2) leaves undefined:
// `__ZN5jaccl12is_availableEv` and `__ZN5jaccl4initEb` (`nm -u libmlx.a | grep jaccl`), which
// `nm` of this object defines verbatim. Verified (sc-25689) by compiling this file with the pinned
// header force-included (`c++ -std=c++17 -I <mlx-src>/mlx/distributed/jaccl/lib -include
// jaccl/jaccl.h -fsyntax-only`): clean, so the declarations agree. Re-check on an MLX bump.
// Below 26.2 MLX compiles `no_jaccl.cpp` instead and nothing references these definitions.
#include <memory>

namespace jaccl {
class Group;
bool is_available() { return false; }
std::shared_ptr<Group> init(bool /*strict*/) { return nullptr; }
} // namespace jaccl
