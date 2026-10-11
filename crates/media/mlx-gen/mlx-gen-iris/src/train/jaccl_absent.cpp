// The JACCL entry points MLX's distributed backend switch references, for an MLX build that ships
// without libjaccl.a (the prebuilt cells): JACCL is unavailable and initializes no group, exactly
// as MLX's own `no_jaccl.cpp` reports when JACCL is compiled out. See build.rs (sc-25689).
#include <memory>

namespace jaccl {
class Group;
bool is_available() { return false; }
std::shared_ptr<Group> init(bool /*strict*/) { return nullptr; }
} // namespace jaccl
