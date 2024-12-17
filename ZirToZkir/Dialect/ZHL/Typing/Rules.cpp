#include "Rules.h"
#include <numeric>

namespace zhl {

using namespace zirgen::Zhl;
using namespace mlir;

mlir::FailureOr<TypeBinding> LiteralTypingRule::
    typeCheck(zirgen::Zhl::LiteralOp op, mlir::ArrayRef<TypeBinding>, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  return getBindings().Const(op.getValue());
}
mlir::FailureOr<TypeBinding> StringTypingRule::
    typeCheck(zirgen::Zhl::StringOp op, mlir::ArrayRef<TypeBinding>, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  return getBindings().Get("String");
}
mlir::FailureOr<TypeBinding> GlobalTypingRule::
    typeCheck(zirgen::Zhl::GlobalOp op, mlir::ArrayRef<TypeBinding>, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  auto binding = getBindings().MaybeGet(op.getName());
  if (mlir::failed(binding)) {
    return op->emitError() << "type '" << op.getName() << "' was not found";
  }
  return binding;
}
mlir::FailureOr<TypeBinding> ParameterTypingRule::
    typeCheck(zirgen::Zhl::ConstructorParamOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  // TODO: Annotation of the component being analyzed with the parameter name and index
  if (operands.empty()) {
    return mlir::failure();
  }
  return op.getVariadic() ? TypeBinding::WrapVariadic(operands[0]) : operands[0];
}
mlir::FailureOr<TypeBinding> ExternTypingRule::
    typeCheck(zirgen::Zhl::ExternOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }
  return operands[0];
}
mlir::FailureOr<TypeBinding> ConstructTypingRule::
    typeCheck(zirgen::Zhl::ConstructOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }
  return operands[0];
}
mlir::FailureOr<TypeBinding> GetGlobalTypingRule::
    typeCheck(zirgen::Zhl::GetGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  // TODO: Add check that the global exists in the scope
  if (operands.empty()) {
    return mlir::failure();
  }
  return operands[0];
}
mlir::FailureOr<TypeBinding> ConstructGlobalTypingRule::
    typeCheck(zirgen::Zhl::ConstructGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  // TODO: Add global declaration to the scope
  if (operands.empty()) {
    return mlir::failure();
  }
  return operands[0];
}
mlir::FailureOr<TypeBinding> SuperTypingRule::
    typeCheck(zirgen::Zhl::SuperOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }
  scope.declareSuperType(operands[0]);
  return operands[0];
}
mlir::FailureOr<TypeBinding> DeclareTypingRule::
    typeCheck(zirgen::Zhl::DeclarationOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    scope.declareMember(op.getMember());
    return getBindings().Bottom();
  } else {
    scope.declareMember(op.getMember(), operands[0]);
    return operands[0];
  }
}

mlir::FailureOr<zirgen::Zhl::DeclarationOp> getDeclaration(zirgen::Zhl::DefinitionOp op) {
  auto decl = op.getDeclaration().getDefiningOp();
  if (!decl) {
    return mlir::failure();
  }
  if (auto declOp = mlir::dyn_cast<zirgen::Zhl::DeclarationOp>(decl)) {
    return declOp;
  }
  return mlir::failure();
}

mlir::FailureOr<TypeBinding> DefineTypeRule::
    typeCheck(zirgen::Zhl::DefinitionOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  auto decl = getDeclaration(op);
  if (mlir::failed(decl)) {
    return op->emitError() << "Malformed IR: Definition must have a declaration operand";
  }
  if (operands.size() < 2) {
    return mlir::failure();
  }
  scope.declareMember(decl->getMember(), operands[1]);
  return operands[1];
}

mlir::FailureOr<TypeBinding> ConstrainTypeRule::
    typeCheck(zirgen::Zhl::ConstraintOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  // TODO: Check that the argument types are correct
  return getBindings().Component();
}
mlir::FailureOr<TypeBinding> GenericParamTypeRule::
    typeCheck(zirgen::Zhl::TypeParamOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }

  scope.declareGenericParam(op.getName(), op.getIndex(), operands[0]);
  return operands[0];
}
mlir::FailureOr<TypeBinding> SpecializeTypeRule::
    typeCheck(zirgen::Zhl::SpecializeOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }
  auto typeToSpecialize = operands[0];
  return typeToSpecialize.specialize([&]() { return op->emitOpError(); }, operands.drop_front());
}
mlir::FailureOr<TypeBinding> SubscriptTypeRule::
    typeCheck(zirgen::Zhl::SubscriptOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }

  return operands[0].getArrayElement([&]() { return op->emitOpError(); });
}
mlir::FailureOr<TypeBinding> ArrayTypeRule::
    typeCheck(zirgen::Zhl::ArrayOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return op->emitOpError() << "could not infer the type of the array";
  }

  auto commonType = std::reduce(
      operands.drop_front().begin(), operands.end(), operands.front(),
      [](auto a, auto b) { return a.commonSupertypeWith(b); }
  );

  return getBindings().Array(commonType, operands.size());
}
mlir::FailureOr<TypeBinding> BackTypeRule::
    typeCheck(zirgen::Zhl::BackOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.size() < 2) {
    return mlir::failure();
  }
  // TODO: Check that distance is a subtype of Val
  return operands[1];
}
mlir::FailureOr<TypeBinding> RangeTypeRule::
    typeCheck(zirgen::Zhl::RangeOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.size() < 2) {
    return mlir::failure();
  }

  auto common = operands[0].commonSupertypeWith(operands[1]);
  return getBindings().UnkArray(common);
}
mlir::FailureOr<TypeBinding> ReduceTypeRule::
    typeCheck(zirgen::Zhl::ReduceOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.size() < 3) {
    return mlir::failure();
  }
  // TODO: Validation that operands[0] is an array
  // TODO: Validation that the inner type of the array is a subtype of the first argument of
  // operands[2]
  // TODO: Validation that the init type is a subtype of the second arguments of operands[2]
  return operands[2];
}
mlir::FailureOr<TypeBinding> ConstructGlobalTypeRule::
    typeCheck(zirgen::Zhl::ConstructGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.size() < 2) {
    return mlir::failure();
  }

  return operands[1];
}

FailureOr<TypeBinding> BlockTypeRule::typeCheck(
    BlockOp op, ArrayRef<TypeBinding> operands, Scope &scope, ArrayRef<const Scope *> regionScopes
) const {
  if (regionScopes.size() != 1) {
    return failure();
  }

  auto super = regionScopes[0]->getSuperType();
  if (failed(super)) {
    return op->emitOpError() << "could not deduce type of block because couldn't get super type";
  }
  return super;
}

FailureOr<TypeBinding> SwitchTypeRule::typeCheck(
    SwitchOp op, ArrayRef<TypeBinding> operands, Scope &scope, ArrayRef<const Scope *> regionScopes
) const {
  if (regionScopes.empty()) {
    return failure();
  }
  return std::transform_reduce(
      regionScopes.begin(), regionScopes.end(), FailureOr(getBindings().Bottom()),
      [](FailureOr<TypeBinding> a, FailureOr<TypeBinding> b) -> FailureOr<TypeBinding> {
    if (failed(a) || failed(b)) {
      return failure();
    }
    return a->commonSupertypeWith(*b);
  }, [](const Scope *scope) { return scope->getSuperType(); }
  );
}

FailureOr<TypeBinding> MapTypeRule::typeCheck(
    MapOp op, ArrayRef<TypeBinding> operands, Scope &scope, ArrayRef<const Scope *> regionScopes
) const {
  if (regionScopes.size() != 1) {
    return op->emitError() << "was expecting to have only one region";
  }

  if (!operands[0].isArray()) {
    return op->emitOpError() << "was expecting a array as input. Got '" << operands[0].getName()
                             << "'";
  }

  assert(!regionScopes.empty());
  auto super = regionScopes[0]->getSuperType();
  if (failed(super)) {
    return op->emitOpError() << "failed to deduce the super type";
  }

  return getBindings().UnkArray(*super);
}

FailureOr<std::vector<TypeBinding>> MapTypeRule::bindRegionArguments(
    ValueRange args, MapOp op, ArrayRef<TypeBinding> operands, Scope &scope
) const {
  if (args.size() != 1) {
    return op->emitError() << "malformed ir: map op must have only one region argument";
  }

  if (!operands[0].isArray()) {
    return op->emitOpError() << "was expecting a array as input. Got '" << operands[0].getName()
                             << "'";
  }

  auto innerType = operands[0].getArrayElement([&]() { return op->emitError(); });
  if (failed(innerType)) {
    return op->emitError() << "failed to extract array inner type";
  }

  return std::vector({*innerType});
}

} // namespace zhl
