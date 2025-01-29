#include "zklang/Dialect/ZHL/Typing/Scope.h"

#include <llvm/Support/raw_ostream.h>

using namespace zirgen::Zhl;
using namespace mlir;

namespace zhl {

ComponentScope::ComponentScope(ComponentOp component, TypeBindings &bindings)
    : bindings(&bindings), component(component) {}

ComponentScope::~ComponentScope() {
  assert(succeeded(superType));
  llvm::dbgs() << "Super type for " << component.getName() << " is ";
  superType->print(llvm::dbgs());
  llvm::dbgs() << "\n";
  bindings->Create(
      component.getName(), component.getLoc(), bindings->Manage(*superType), genericParams,
      constructorParams, members
  );
}

void ComponentScope::declareGenericParam(StringRef name, uint64_t index, TypeBinding type) {
  genericParams.insert({{name, index}, type});
}

void ComponentScope::declareConstructorParam(StringRef name, uint64_t index, TypeBinding type) {
  constructorParams.insert({{name, index}, type});
}

void ComponentScope::declareMember(StringRef name) {
  assert(members.find(name) == members.end());
  members[name] = std::nullopt;
}

void ComponentScope::declareSuperType(TypeBinding type) { superType = type; }

void ComponentScope::declareMember(StringRef name, TypeBinding type) {
  // If the value is present and we are (re-)declaring it
  // we can only do so if it has not value.
  auto it = members.find(name);
  if (it != members.end() && it->second.has_value()) {
    return;
  }
  members[name] = type;
}

bool ComponentScope::memberDeclaredWithType(StringRef name) {
  auto it = members.find(name);
  return it != members.end() && it->second.has_value();
}

ComponentOp ComponentScope::getOp() const { return component; }
FailureOr<TypeBinding> ComponentScope::getSuperType() const { return superType; }

TypeBindings &ComponentScope::getTypeBindings() { return *bindings; }

/* BlockScope */

BlockScope::BlockScope(Scope &parent, mlir::Region &region) : parent(&parent), region(region) { assert(this->parent != nullptr); }

BlockScope::~BlockScope() {
  if (!shadowedMembers.empty()) {
    auto &bindings = getTypeBindings();
    std::string regionName;
    llvm::raw_string_ostream ss(regionName);
    region.getLoc().print(ss);
    bindings.Create(
        regionName, region.getLoc(), bindings.Manage(*superType), /* generic */ ParamsMap(),
        /* constructor */ ParamsMap(), shadowedMembers
    );
  }
}

void BlockScope::declareGenericParam(StringRef name, uint64_t index, TypeBinding type) {
  parent->declareGenericParam(name, index, type);
}

void BlockScope::declareConstructorParam(StringRef name, uint64_t index, TypeBinding type) {
  parent->declareConstructorParam(name, index, type);
}

void BlockScope::declareMember(StringRef name) {
  assert(shadowedMembers.find(name) == shadowedMembers.end());
  shadowedMembers[name] = std::nullopt;
}

void BlockScope::declareSuperType(TypeBinding type) { superType = type; }

void BlockScope::declareMember(StringRef name, TypeBinding type) {
  // If the value is present and we are (re-)declaring it
  // we can only do so if it has not value.
  auto it = shadowedMembers.find(name);
  if (it != shadowedMembers.end() && it->second.has_value()) {
    return;
  }
  shadowedMembers[name] = type;
}

bool BlockScope::memberDeclaredWithType(StringRef name) {
  auto it = shadowedMembers.find(name);
  if (it != shadowedMembers.end() && it->second.has_value()) {
    return true;
  }
  return parent->memberDeclaredWithType(name);
}
ComponentOp BlockScope::getOp() const { return parent->getOp(); }

FailureOr<TypeBinding> BlockScope::getSuperType() const { return superType; }

TypeBindings &BlockScope::getTypeBindings() { return parent->getTypeBindings(); }

} // namespace zhl
