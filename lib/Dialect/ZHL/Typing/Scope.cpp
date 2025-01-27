#include "zklang/Dialect/ZHL/Typing/Scope.h"

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
  if (members.find(name) != members.end() && members[name].has_value()) {
    return;
  }
  members[name] = type;
}

bool ComponentScope::memberDeclaredWithType(StringRef name) { return members[name].has_value(); }

ComponentOp ComponentScope::getOp() const { return component; }
FailureOr<TypeBinding> ComponentScope::getSuperType() const { return superType; }

BlockScope::BlockScope(Scope &parent) : parent(&parent) { assert(this->parent != nullptr); }

void BlockScope::declareGenericParam(StringRef name, uint64_t index, TypeBinding type) {
  parent->declareGenericParam(name, index, type);
}

void BlockScope::declareConstructorParam(StringRef name, uint64_t index, TypeBinding type) {
  parent->declareConstructorParam(name, index, type);
}

void BlockScope::declareMember(StringRef name) { parent->declareMember(name); }

void BlockScope::declareSuperType(TypeBinding type) { superType = type; }

void BlockScope::declareMember(StringRef name, TypeBinding type) {
  parent->declareMember(name, type);
}

bool BlockScope::memberDeclaredWithType(StringRef name) {
  return parent->memberDeclaredWithType(name);
}
ComponentOp BlockScope::getOp() const { return parent->getOp(); }

FailureOr<TypeBinding> BlockScope::getSuperType() const { return superType; }

} // namespace zhl
