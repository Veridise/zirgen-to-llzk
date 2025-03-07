#include <cassert>
#include <cstdint>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/Scope.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

using namespace zirgen::Zhl;
using namespace mlir;

namespace zhl {

ComponentScope::ComponentScope(ComponentOp component, TypeBindings &bindings)
    : Scope(Sco_Component), bindings(&bindings), component(component) {}

ComponentScope::~ComponentScope() { createBinding(component.getName(), component.getLoc()); }

void ComponentScope::declareGenericParam(StringRef name, uint64_t index, TypeBinding type) {
  genericParams.insert({name, {type, index}});
}

TypeBinding ComponentScope::declareLiftedAffineToGenericParam(const TypeBinding &type) {
  assert(type.hasConstExpr() && "Can't lift a non constant binding");
  assert(
      mlir::isa<expr::CtorExpr>(type.getConstExpr()) &&
      "Can't lift a constant binding that is not a constructor expression"
  );

  Twine name("Aff$" + Twine(liftedParams.size()));
  auto allocName = name.str();
  liftedParams.insert({allocName, {type, liftedParams.size()}});
  return TypeBinding::MakeGenericParam(bindings->Manage(type), allocName);
}

void ComponentScope::declareConstructorParam(StringRef name, uint64_t index, TypeBinding type) {
  constructorParams.insert({name, {type, index}});
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

size_t ComponentScope::memberCount() const { return members.size(); }
Frame &ComponentScope::getCurrentFrame() { return frame; }

const Frame &ComponentScope::getCurrentFrame() const { return frame; }

FailureOr<TypeBinding> ComponentScope::getSuperType() const { return superType; }

TypeBinding ComponentScope::createBinding(StringRef name, Location loc) const {
  assert(succeeded(superType));
  return bindings->Create(
      name, loc, bindings->Manage(*superType), genericParams, constructorParams, members, frame
  );
}

ChildScope::ChildScope(Scope &parentScope) : ChildScope(Sco_Child, parentScope) {}

ChildScope::ChildScope(ScopeKind Kind, Scope &parentScope) : Scope(Kind), parent(&parentScope) {
  assert(parent != nullptr);
}

void ChildScope::declareGenericParam(StringRef name, uint64_t index, TypeBinding type) {
  parent->declareGenericParam(name, index, type);
}

TypeBinding ChildScope::declareLiftedAffineToGenericParam(const TypeBinding &type) {
  return parent->declareLiftedAffineToGenericParam(type);
}

void ChildScope::declareConstructorParam(StringRef name, uint64_t index, TypeBinding type) {
  parent->declareConstructorParam(name, index, type);
}

void ChildScope::declareMember(mlir::StringRef name) { parent->declareMember(name); }

void ChildScope::declareMember(mlir::StringRef name, TypeBinding type) {
  parent->declareMember(name, type);
}

size_t ChildScope::memberCount() const { return parent->memberCount(); }
bool ChildScope::memberDeclaredWithType(mlir::StringRef name) {
  return parent->memberDeclaredWithType(name);
}

void ChildScope::declareSuperType(TypeBinding type) { parent->declareSuperType(type); }

TypeBinding ChildScope::createBinding(StringRef name, Location loc) const {
  return parent->createBinding(name, loc);
}

zirgen::Zhl::ComponentOp ChildScope::getOp() const { return parent->getOp(); }

mlir::FailureOr<TypeBinding> ChildScope::getSuperType() const { return parent->getSuperType(); }

Frame &ChildScope::getCurrentFrame() { return parent->getCurrentFrame(); }
const Frame &ChildScope::getCurrentFrame() const { return parent->getCurrentFrame(); }

FrameScope::FrameScope(Scope &parent, Frame frame) : ChildScope(Sco_Frame, parent), frame(frame) {}

Frame &FrameScope::getCurrentFrame() { return frame; }
const Frame &FrameScope::getCurrentFrame() const { return frame; }

BlockScope::BlockScope(Scope &parent, TypeBindings &Bindings)
    : ChildScope(Sco_Block, parent), bindings(&Bindings) {}

void BlockScope::declareSuperType(TypeBinding type) { superType = type; }

void BlockScope::declareMember(StringRef name) { declareMemberImpl(name); }

void BlockScope::declareMember(StringRef name, TypeBinding type) { declareMemberImpl(name, type); }

bool BlockScope::memberDeclaredWithType(StringRef name) { return memberDeclaredWithTypeImpl(name); }

size_t BlockScope::memberCount() const { return members.size(); }

void collectTypeVarsFromGenericParams(
    const TypeBinding &binding, llvm::StringMap<const TypeBinding *> &varNames
) {
  for (auto &param : binding.getGenericParams()) {
    if (param.isGenericParam()) {
      varNames.insert({param.getGenericParamName(), &param});
    } else {
      collectTypeVarsFromGenericParams(param, varNames);
    }
  }
}

TypeBinding BlockScope::createBinding(StringRef name, Location loc) const {
  assert(succeeded(superType));
  ParamsMap ctorArgs({{"super", {*superType, 0}}});
  std::vector<std::string_view> sortedFieldNames;
  sortedFieldNames.reserve(members.size());
  llvm::StringMap<const TypeBinding *> varNames;

  std::transform(
      members.begin(), members.end(), std::back_inserter(sortedFieldNames),
      [&](auto &p) {
    collectTypeVarsFromGenericParams(*p.second, varNames);
    return p.first();
  }
  );
  std::sort(sortedFieldNames.begin(), sortedFieldNames.end());
  size_t argNo = 1;
  for (auto &fieldName : sortedFieldNames) {
    auto memberBinding = members.at(fieldName);
    assert(memberBinding.has_value());
    ctorArgs.insert({fieldName, {*memberBinding, argNo}});
    argNo++;
  }
  collectTypeVarsFromGenericParams(*superType, varNames);
  ParamsMap genericParams;
  size_t paramNo = 0;
  for (auto &[varName, type] : varNames) {
    genericParams.insert({varName, {*type, paramNo}});
    paramNo++;
  }

  return TypeBinding::WithClosure(bindings->CreateAnon(
      name, loc, bindings->Manage(*superType), genericParams, ctorArgs, members, getCurrentFrame()
  ));
}

FailureOr<TypeBinding> BlockScope::getSuperType() const { return superType; }

void LexicalScopeImpl::declareMemberImpl(StringRef name) {
  assert(members.find(name) == members.end());
  members[name] = std::nullopt;
}

void LexicalScopeImpl::declareMemberImpl(StringRef name, TypeBinding type) {
  // If the value is present and we are (re-)declaring it
  // we can only do so if it has not value.
  if (!memberDeclaredWithTypeImpl(name)) {
    members[name] = type;
  }
}

bool LexicalScopeImpl::memberDeclaredWithTypeImpl(StringRef name) {
  return members[name].has_value();
}

} // namespace zhl
