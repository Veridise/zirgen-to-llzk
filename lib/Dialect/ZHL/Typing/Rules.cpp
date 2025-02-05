#include "zklang/Dialect/ZHL/Typing/Rules.h"
#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include <mlir/Support/LogicalResult.h>
#include <numeric>
#include <zklang/Dialect/ZHL/Typing/ArrayFrame.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameImpl.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/InnerFrame.h>

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
  llvm::dbgs() << "Typechecking " << op << "\n";
  auto binding = getBindings().MaybeGet(op.getName());
  if (mlir::failed(binding)) {
    llvm::dbgs() << "Failed to obtain a binding for " << op.getName() << "\n";
    return op->emitError() << "type '" << op.getName() << "' was not found";
  }
  llvm::dbgs() << "Found binding for " << op.getName() << ": ";
  binding->print(llvm::dbgs());
  llvm::dbgs() << "\n";
  return binding;
}
mlir::FailureOr<TypeBinding> ParameterTypingRule::
    typeCheck(zirgen::Zhl::ConstructorParamOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }
  auto arg = (op.getVariadic() ? TypeBinding::WrapVariadic(operands[0]) : operands[0])
                 .WithUpdatedLocation(op.getLoc());
  scope.declareConstructorParam(op.getName(), op.getIndex(), arg);
  return arg;
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

  // TODO: Only do this for the constructor calls that will not be lowered to llzk ops
  //       Meaning, any builtin that was not overriden and that will lower to a llzk operation that
  //       is not ComputeOnly don't need to allocate a frame. This will avoid creating unnecessary
  //       fields.
  auto component = operands[0];
  scope.getCurrentFrame().allocateSlot<ComponentSlot>(getBindings(), component);
  return component;
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
  }
  scope.declareMember(op.getMember(), operands[0]);
  auto binding = operands[0]; // Make a copy for marking the slot
  scope.getCurrentFrame().allocateSlot<ComponentSlot>(
      getBindings(), binding, op.getMember()
  ); // Allocate a named slot with the declared type
  return binding;
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

/// Defines the value of a member inside a component.
/// This op takes two operands, the first operand must always be a declaration that refers to the
/// member whose value is being defined. The second is any value that is written into the member.
/// The member can be declared with a type instead of using the type of the expression.
///
/// The former looks as follows
/// ```
/// x : Val;
/// x := 1;
/// ```
///
/// While the laller as follows
/// ```
/// x := 1;
/// ```
///
/// These two snippets of code are equivalent. The difference comes when a subtype of the declared
/// type is used in the expression. For example, these two expressions result in different types.
///
/// In this case `x` is of type `Val`
/// ```
/// x : Val
/// x := 2 * 10;
/// ```
///
/// While in this case is of type `Mul` since `2 * 10` is actually `Mul(2,10)`.
/// ```
/// x := 2 * 10;
/// ```
///
/// Therefore it is necessary to preserve and respect the type declared by `DeclarationOp`.
///
/// On the opposite side, the type generated by the expression matters because it could generate
/// constrains. For example the following snippet will generate a constrain via the `Reg` component.
/// ```
/// x : Val;
/// x := Reg(10);
/// ```
///
/// Since `Reg` is a subtype of `Val` there is an implicit conversion to `Val` and the code is
/// properly typed. The lowering steps however need to take into consideration the possibility of
/// the expression generating constraints. For this reason, all constructor calls via the
/// `ConstructOp` allocate a temporary slot with the called component type. In the example above
/// `Reg` would be written into a temporary before its super-super type gets stored in `x`. Failing
/// to consider this means that the type of the expression gets erased and it becomes impossible to
/// call the appropiate constrain function.
///
/// If the value of the expression comes from another op that does not allocate a temporary (i.e. a
/// literal Val expression) then the binding of the expression will not have a slot associated
/// with it.
///
/// This leaves us with 5 cases that need to be considered:
///
/// 1. Both operands allocated a slot and have different type bindings: In this case the result of
/// this rule is a copy of the type binding of the declaration op. The slot allocated by the
/// expression needs to be maintained to avoid losing the type information.
///
/// Op responsible of instantiating the slot while lowering:
///  - Declaration operation: For the named slot
///  - Expression operation: For its own slot
///
/// 2. Both operands allocated a slot and have the same type binding: The result is a copy of the
/// type binding of the expression op. The slot of the expression op is renamed to the member's
/// name. The type binding returned by this rule does not link to a slot since the expression op
/// already does.
///
/// Op responsible of instantiating the slot while lowering:
///  - Expression operation
///
/// 3. The declaration allocated a slot but the expression didn't: The result is a copy of the type
/// binding of the declaration op.
///
/// Op responsible of instantiating the slot while lowering:
///  - Declaration operation
///
/// 4. The expression allocated a slot but the declaration didn't: Rename the slot with the name of
/// the member.
///
/// Op responsible of instantiating the slot while lowering:
///  - Expression operation
///
/// 5. Neither operation allocated a slot: Allocate a slot of the type binding of the expression and
/// the name of the member. Return a type binding that links to the allocated slot.
///
/// Op responsible of instantiating the slot while lowering:
///  - This operation: Since the slot is allocated during the execution of this rule.
///
mlir::FailureOr<TypeBinding> DefineTypeRule::
    typeCheck(zirgen::Zhl::DefinitionOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  auto copyWithoutSlot = [](const TypeBinding &binding) {
    auto copy = binding;
    copy.markSlot(nullptr);
    return copy;
  };

  auto decl = getDeclaration(op);
  if (mlir::failed(decl)) {
    return op->emitError() << "Malformed IR: Definition must have a declaration operand";
  }
  if (operands.size() < 2) {
    return mlir::failure();
  }
  if (!operands[0].isBottom() && failed(operands[1].subtypeOf(operands[0]))) {
    return op->emitError() << "was expecting a subtype of '" << operands[0] << "', but got '"
                           << operands[1] << "'";
  }

  auto *declSlot = operands[0].getSlot();
  auto *exprSlot = operands[1].getSlot();

  // Case 1 & 2
  if (declSlot && exprSlot) {
    auto *declCompSlot = mlir::cast<ComponentSlot>(declSlot);
    auto *exprCompSlot = mlir::cast<ComponentSlot>(exprSlot);
    auto declBinding = declCompSlot->getBinding();
    auto exprBinding = exprCompSlot->getBinding();

    // Case 2: The result is a copy of the type binding of the expression op. The slot of the
    // expression op is renamed to the member's name. The type binding returned by
    // this rule does not link to a slot since the expression op already does.
    // FIXME: Change it to proper equality after merging since the implementation for that is
    // already in the main branch
    if (declBinding.getName() == exprBinding.getName()) {
      exprSlot->rename(decl->getMember());
      scope.declareMember(decl->getMember(), operands[1]);
      return copyWithoutSlot(operands[1]);
    }

    // Case 1: In this case the result of this rule is a copy of the type binding of the declaration
    // op. The slot allocated by the expression needs to be maintained to avoid losing the type
    // information.
    return copyWithoutSlot(operands[0]);
  }

  // Case 3: The result is a copy of the type binding of the declaration op.
  if (declSlot) {
    return copyWithoutSlot(operands[0]);
  }

  // Case 4: Rename the slot with the name of the member.
  if (exprSlot) {
    exprSlot->rename(decl->getMember());
    scope.declareMember(decl->getMember(), operands[1]);
    return copyWithoutSlot(operands[1]);
  }

  // Case 5: Allocate a slot of the type binding of the expression and the name of the member.
  // Return a type binding that links to the allocated slot.
  auto binding = operands[1];
  scope.getCurrentFrame().allocateSlot<ComponentSlot>(getBindings(), binding, decl->getMember());
  scope.declareMember(decl->getMember(), binding);
  return binding;
}

mlir::FailureOr<TypeBinding> ConstrainTypeRule::
    typeCheck(zirgen::Zhl::ConstraintOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  // TODO: Check that the argument types are correct
  // XXX: Improvement idea; return the least common super type of the arguments instead of
  // `Component`.
  return getBindings().Component();
}

mlir::FailureOr<TypeBinding> GenericParamTypeRule::
    typeCheck(zirgen::Zhl::TypeParamOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }

  scope.declareGenericParam(op.getName(), op.getIndex(), operands[0]);
  return TypeBinding::MakeGenericParam(getBindings().Manage(operands[0]), op.getName());
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
      [](auto a, auto b) {
    // llvm::dbgs() << "a = ";
    // a.print(llvm::dbgs());
    // llvm::dbgs() << ", b = ";
    // b.print(llvm::dbgs());
    // llvm::dbgs() << "\n";
    return a.commonSupertypeWith(b);
  }
  );
  // llvm::dbgs() << "final common type = ";
  // commonType.print(llvm::dbgs());
  // llvm::dbgs() << "\n";

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
  // TODO: The reduce op needs to create an array frame inside the current frame
  //       where it can hold the intermediate results of the reduce loop.
  //       This way if the accumulation operation is a complex component with constrains the loop
  //       will be able to call the constrain call of each intermediate step.
  return operands[1];
}

mlir::FailureOr<Frame> ReduceTypeRule::allocate(Frame frame) const {
  return frame.allocateSlot<ArrayFrame>()->getFrame();
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

  // TODO: Allocate a frame inside the current frame
  auto super = regionScopes[0]->getSuperType();
  if (failed(super)) {
    return op->emitOpError() << "could not deduce type of block because couldn't get super type";
  }
  return super;
}

FailureOr<Frame> BlockTypeRule::allocate(Frame frame) const {
  return frame.allocateSlot<InnerFrame>()->getFrame();
}

FailureOr<TypeBinding> SwitchTypeRule::typeCheck(
    SwitchOp op, ArrayRef<TypeBinding> operands, Scope &scope, ArrayRef<const Scope *> regionScopes
) const {
  if (regionScopes.empty()) {
    return failure();
  }
  // TODO: Allocate a frame for the switch arms
  return std::transform_reduce(
      regionScopes.begin(), regionScopes.end(), FailureOr<TypeBinding>(getBindings().Bottom()),
      [](FailureOr<TypeBinding> a, FailureOr<TypeBinding> b) -> FailureOr<TypeBinding> {
    if (failed(a) || failed(b)) {
      return failure();
    }
    return a->commonSupertypeWith(*b);
  }, [](const Scope *armScope) { return armScope->getSuperType(); }
  );
}

FailureOr<Frame> SwitchTypeRule::allocate(Frame) const {
  return failure(); // TODO
}

FailureOr<TypeBinding> MapTypeRule::typeCheck(
    MapOp op, ArrayRef<TypeBinding> operands, Scope &scope, ArrayRef<const Scope *> regionScopes
) const {
  // TODO: Allocate an array frame for holding all the stuff happening inside the loop
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

FailureOr<Frame> MapTypeRule::allocate(Frame frame) const {
  return frame.allocateSlot<ArrayFrame>()->getFrame();
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

FailureOr<TypeBinding> LookupTypeRule::
    typeCheck(LookupOp op, ArrayRef<TypeBinding> operands, Scope &scope, ArrayRef<const Scope *>)
        const {
  // Get the type of the component argument
  if (operands.empty()) {
    return failure();
  }
  auto &comp = operands[0];

  return comp.getMember(op.getMember(), [&]() { return op->emitError(); });
}

} // namespace zhl
