#include <cassert>
#include <cstdint>
#include <llvm/ADT/StringRef.h>
#include <zklang/Dialect/ZHL/Typing/Scope.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

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
      constructorParams, members, frame
  );
}

void ComponentScope::declareGenericParam(StringRef name, uint64_t index, TypeBinding type) {
  genericParams.insert({{name, index}, type});
}

void ComponentScope::declareConstructorParam(StringRef name, uint64_t index, TypeBinding type) {
  constructorParams.insert({{name, index}, type});
}

void ComponentScope::declareSuperType(TypeBinding type) { superType = type; }

ComponentOp ComponentScope::getOp() const { return component; }

void ComponentScope::declareMember(StringRef name) { declareMemberImpl(name); }

void ComponentScope::declareMember(StringRef name, TypeBinding type) {
  declareMemberImpl(name, type);
}

bool ComponentScope::memberDeclaredWithType(StringRef name) {
  return memberDeclaredWithTypeImpl(name);
}

Frame &ComponentScope::getCurrentFrame() { return frame; }

FailureOr<TypeBinding> ComponentScope::getSuperType() const { return superType; }

ChildScope::ChildScope(Scope &parentScope) : parent(&parentScope) { assert(parent != nullptr); }

void ChildScope::declareGenericParam(StringRef name, uint64_t index, TypeBinding type) {
  parent->declareGenericParam(name, index, type);
}

void ChildScope::declareConstructorParam(StringRef name, uint64_t index, TypeBinding type) {
  parent->declareConstructorParam(name, index, type);
}

void ChildScope::declareMember(mlir::StringRef name) { parent->declareMember(name); }

void ChildScope::declareMember(mlir::StringRef name, TypeBinding type) {
  parent->declareMember(name, type);
}

bool ChildScope::memberDeclaredWithType(mlir::StringRef name) {
  return parent->memberDeclaredWithType(name);
}

void ChildScope::declareSuperType(TypeBinding type) { parent->declareSuperType(type); }

zirgen::Zhl::ComponentOp ChildScope::getOp() const { return parent->getOp(); }

mlir::FailureOr<TypeBinding> ChildScope::getSuperType() const { return parent->getSuperType(); }

Frame &ChildScope::getCurrentFrame() { return parent->getCurrentFrame(); }

FrameScope::FrameScope(Scope &parent, Frame frame) : ChildScope(parent), frame(frame) {}

Frame &FrameScope::getCurrentFrame() { return frame; }

BlockScope::BlockScope(Scope &parent) : ChildScope(parent) {}

void BlockScope::declareSuperType(TypeBinding type) { superType = type; }

void BlockScope::declareMember(StringRef name) { declareMemberImpl(name); }

void BlockScope::declareMember(StringRef name, TypeBinding type) { declareMemberImpl(name, type); }

bool BlockScope::memberDeclaredWithType(StringRef name) { return memberDeclaredWithTypeImpl(name); }

FailureOr<TypeBinding> BlockScope::getSuperType() const { return superType; }

void LexicalScopeImpl::declareMemberImpl(StringRef name) {
  assert(members.find(name) == members.end());
  members[name] = std::nullopt;
}

void LexicalScopeImpl::declareMemberImpl(StringRef name, TypeBinding type) {
  // If the value is present and we are (re-)declaring it
  // we can only do so if it has not value.
  if (members.find(name) != members.end() && members[name].has_value()) {
    return;
  }
  members[name] = type;
}

bool LexicalScopeImpl::memberDeclaredWithTypeImpl(StringRef name) {
  return members[name].has_value();
}

} // namespace zhl
