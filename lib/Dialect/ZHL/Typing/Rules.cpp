#include <mlir/Support/LogicalResult.h>
#include <numeric>
#include <zklang/Dialect/ZHL/Typing/ArrayFrame.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameImpl.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/InnerFrame.h>
#include <zklang/Dialect/ZHL/Typing/Interpreter.h>
#include <zklang/Dialect/ZHL/Typing/Rules.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>

#define DEBUG_TYPE "type-checker"

namespace zhl {

using namespace zirgen::Zhl;
using namespace mlir;
using namespace expr::interpreter;

mlir::FailureOr<TypeBinding> LiteralTypingRule::
    typeCheck(zirgen::Zhl::LiteralOp op, mlir::ArrayRef<TypeBinding>, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  return interpretateOp(op, getBindings().Const(op.getValue()));
}

mlir::FailureOr<TypeBinding> StringTypingRule::
    typeCheck(zirgen::Zhl::StringOp op, mlir::ArrayRef<TypeBinding>, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  return interpretateOp(op, getBindings().Get("String"));
}

mlir::FailureOr<TypeBinding> GlobalTypingRule::
    typeCheck(zirgen::Zhl::GlobalOp op, mlir::ArrayRef<TypeBinding>, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  auto binding = getBindings().MaybeGet(op.getName());
  if (mlir::failed(binding)) {
    return op->emitError() << "type '" << op.getName() << "' was not found";
  }
  return interpretateOp(op, *binding);
}

// If the type we are going to declare as constructor parameter has a parameter that would
// generate an AffineMap (i.e. a CtorExpr) then we declare a new generic parameter that has the
// type of the binding that would generate the AffineMap and replace it in the type with a
// variable that points to the binding.

static void liftCtorExpressions(TypeBinding &binding, Scope &scope) {
  if (mlir::isa<expr::CtorExpr>(binding.getConstExpr())) {
    binding = scope.declareLiftedAffineToGenericParam(binding);
    return;
  }

  assert(binding.isSpecialized());
  for (auto &param : binding.getGenericParams()) {
    liftCtorExpressions(param, scope);
  }
}

mlir::FailureOr<TypeBinding> ParameterTypingRule::
    typeCheck(zirgen::Zhl::ConstructorParamOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }
  auto &arg = operands[0];
  if (!arg.isSpecialized()) {
    return op->emitError() << "cannot use on non-concrete type '" << arg
                           << "' as component argument #" << op.getIndex();
  }
  if (arg.hasConstExpr()) {
    return op->emitError() << "cannot use a constant expression as parameter type";
  }
  auto interpretatedArg = interpretateOp(
      op, (op.getVariadic() ? TypeBinding::WrapVariadic(arg) : arg).WithUpdatedLocation(op.getLoc())
  );
  liftCtorExpressions(interpretatedArg, scope);
  scope.declareConstructorParam(op.getName(), op.getIndex(), interpretatedArg);
  return interpretatedArg;
}

mlir::FailureOr<TypeBinding> ExternTypingRule::
    typeCheck(zirgen::Zhl::ExternOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }
  return interpretateOp(op, operands[0]);
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
  if (operands[0].isBuiltin() && zml::isBuiltinDontNeedAlloc(operands[0].getName())) {
    return interpretateOp(op, operands[0], operands.drop_front());
  }
  auto component = interpretateOp(op, operands[0], operands.drop_front());
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
  return interpretateOp(op, operands[0]);
}

mlir::FailureOr<TypeBinding> ConstructGlobalTypingRule::
    typeCheck(zirgen::Zhl::ConstructGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  // TODO: Add global declaration to the scope
  if (operands.empty()) {
    return mlir::failure();
  }
  return interpretateOp(op, operands[0]);
}

mlir::FailureOr<TypeBinding> SuperTypingRule::
    typeCheck(zirgen::Zhl::SuperOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }
  auto super = interpretateOp(op, operands[0]);
  scope.declareSuperType(super);
  // If this super is the terminator of a block we create a type that represents it.
  if (mlir::isa<BlockOp>(op->getParentOp()) && scope.memberCount() > 0) {
    auto blockBinding = scope.createBinding("block$", op->getParentOp()->getLoc());
    scope.getCurrentFrame().allocateSlot<ComponentSlot>(getBindings(), blockBinding);
    scope.declareSuperType(blockBinding); // Override supertype with the binding we created.
    return blockBinding;
  }

  return super;
}

mlir::FailureOr<TypeBinding> DeclareTypingRule::
    typeCheck(zirgen::Zhl::DeclarationOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    scope.declareMember(op.getMember());
    return interpretateOp(op, getBindings().Bottom());
  }
  auto binding = interpretateOp(op, operands[0]); // Make a copy for marking the slot
  scope.getCurrentFrame().allocateSlot<ComponentSlot>(
      getBindings(), binding, op.getMember()
  ); // Allocate a named slot with the declared type
  scope.declareMember(op.getMember(), binding);
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
/// While the latter as follows
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
/// constraints. For example the following snippet will generate a constraint via the `Reg`
/// component.
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
/// Another thing to consider is if the slot allocated by the expression is from the current frame
/// or not. It could happen that a member is declared by reading some other member:
///
/// ```
/// x := foo.bar
/// ```
///
/// Simply renaming the slot from the expression will actually change the name of the member from
/// 'bar' to 'x'.
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

  LLVM_DEBUG(llvm::dbgs() << "[DefinitionOp rule] Type checking op " << op << "\n");
  auto decl = getDeclaration(op);
  if (mlir::failed(decl)) {
    return op->emitError() << "malformed IR: Definition must have a declaration operand";
  }
  if (operands.size() < 2) {
    return op->emitError() << "not enough arguments for define op";
  }
  if (!operands[0].isBottom() && failed(operands[1].subtypeOf(operands[0]))) {
    return op->emitError() << "was expecting a subtype of '" << operands[0] << "', but got '"
                           << operands[1] << "'";
  }

  auto interpretatedOperand0 = interpretateOp(op, operands[0]);
  auto interpretatedOperand1 = interpretateOp(op, operands[1]);

  LLVM_DEBUG(llvm::dbgs() << "[DefinitionOp rule] Validation passed\n");
  auto *declSlot = operands[0].getSlot();
  auto *exprSlot = operands[1].getSlot();

  // For cases 2 & 4 we want to reuse the slot used by the expression binding. However, this is only
  // safe to do if the frame of that binding belongs to the current frame. If that's not the case
  // then we need to allocate a new frame like in Case 5.
  auto maybeUpdateExprSlot = [&](const TypeBinding &binding) -> TypeBinding {
    assert(exprSlot);
    if (exprSlot->belongsTo(scope.getCurrentFrame())) {
      LLVM_DEBUG(
          llvm::dbgs() << "Renaming slot " << exprSlot << " to '" << decl->getMember() << "'\n"
      );
      exprSlot->rename(decl->getMember());
      return copyWithoutSlot(binding);
    } else {
      LLVM_DEBUG(
          llvm::dbgs() << "Allocating a new slot '" << decl->getMember() << "' for type " << binding
                       << "\n"
      );
      auto copy = copyWithoutSlot(binding);
      auto *slot = scope.getCurrentFrame().allocateSlot<ComponentSlot>(
          getBindings(), copy, decl->getMember()
      );
      (void)slot;
      LLVM_DEBUG(llvm::dbgs() << "Created a new slot " << slot << "\n");
      return copy;
    }
  };

  // Case 1 & 2
  if (declSlot && exprSlot) {
    auto *declCompSlot = mlir::cast<ComponentSlot>(declSlot);
    auto *exprCompSlot = mlir::cast<ComponentSlot>(exprSlot);
    auto declBinding = declCompSlot->getBinding();
    auto exprBinding = exprCompSlot->getBinding();

    // Case 2: The result is a copy of the type binding of the expression op. The slot of the
    // expression op is renamed to the member's name. The type binding returned by
    // this rule does not link to a slot since the expression op already does.
    if (declBinding == exprBinding) {
      LLVM_DEBUG(llvm::dbgs() << "[DefinitionOp rule] Case 2\n");
      scope.declareMember(decl->getMember(), interpretatedOperand1);
      return maybeUpdateExprSlot(interpretatedOperand1);
    }

    // Case 1: In this case the result of this rule is a copy of the type binding of the declaration
    // op. The slot allocated by the expression needs to be maintained to avoid losing the type
    // information.
    LLVM_DEBUG(llvm::dbgs() << "[DefinitionOp rule] Case 1\n");
    return copyWithoutSlot(interpretatedOperand0);
  }

  // Case 3: The result is a copy of the type binding of the declaration op.
  if (declSlot) {
    LLVM_DEBUG(llvm::dbgs() << "[DefinitionOp rule] Case 3\n");
    return copyWithoutSlot(interpretatedOperand0);
  }

  // Case 4: Rename the slot with the name of the member.
  if (exprSlot) {
    LLVM_DEBUG(llvm::dbgs() << "[DefinitionOp rule] Case 4\n");
    scope.declareMember(decl->getMember(), interpretatedOperand1);
    return maybeUpdateExprSlot(interpretatedOperand1);
  }

  // Case 5: Allocate a slot of the type binding of the expression and the name of the member.
  // Return a type binding that links to the allocated slot.
  LLVM_DEBUG(llvm::dbgs() << "[DefinitionOp rule] Case 5\n");
  auto binding = interpretatedOperand1;
  scope.getCurrentFrame().allocateSlot<ComponentSlot>(getBindings(), binding, decl->getMember());
  scope.declareMember(decl->getMember(), binding);
  return binding;
}

mlir::FailureOr<TypeBinding> ConstrainTypeRule::
    typeCheck(zirgen::Zhl::ConstraintOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.size() != 2) {
    return op.emitError() << "constraint expression expects 2 operands, but got "
                          << operands.size();
  }
  auto &lhs = operands[0];
  auto &rhs = operands[1];

  auto leastCommon = lhs.commonSupertypeWith(rhs);
  if (leastCommon.isArray()) {
    auto leastCommonArray = leastCommon.getConcreteArrayType();
    if (mlir::failed(leastCommonArray)) {
      return op.emitError()
             << "constraint operands have Array supertype but it could not be deduced";
    }
    return interpretateOp(op, *leastCommonArray);
  }
  auto &Val = getBindings().Get("Val");
  if (succeeded(leastCommon.subtypeOf(Val))) {
    return interpretateOp(op, Val);
  }

  return interpretateOp(op, getBindings().Component());
}

mlir::FailureOr<TypeBinding> GenericParamTypeRule::
    typeCheck(zirgen::Zhl::TypeParamOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }

  auto param = interpretateOp(
      op, TypeBinding::MakeGenericParam(getBindings().Manage(operands[0]), op.getName())
  );
  scope.declareGenericParam(op.getName(), op.getIndex(), param);
  return param;
}

namespace {

/// Checks that the source value and type binding are valid for the parameter they are going to
/// specialize.
inline LogicalResult checkSpecializationArg(
    Value value, const TypeBinding &type, const TypeBinding &paramType,
    llvm::function_ref<InFlightDiagnostic()> emitError
) {
  if (!paramType.isGenericParam()) {
    return emitError() << "non generic param type";
  }
  auto &paramSuperType = paramType.getSuperType();
  bool paramIsVal = paramSuperType.isVal();
  bool paramIsType = paramSuperType.isTypeMarker();
  if (!(paramIsVal || paramIsType)) {
    return emitError() << "generic param that is neither 'Val' nor 'Type': '" << paramType
                       << "' (super type: '" << paramSuperType << "')";
  }
  // The Value must come from a closed set of operations
  auto op = value.getDefiningOp();
  if (!mlir::isa_and_present<TypeParamOp, LiteralOp, SpecializeOp, GlobalOp, ConstructOp>(op)) {
    return emitError() << "was expecting a type, literal value, constant expression or generic "
                          "parameter, but got a value";
  }
  // If it comes from a TypeParamOp depends on the type of the generic param.
  //  If it's Type then TypeParamOp's binding must not have a ConstExpr
  //  If it's Val then TypeParamOp's binding must have a ConstExpr
  if (mlir::isa<TypeParamOp>(op)) {
    auto hasCE = type.hasConstExpr();
    if (paramIsVal && hasCE) {
      return success();
    }
    if (paramIsVal && !hasCE) {
      return emitError() << "parameter '" << paramType
                         << "' of type 'Val' expects a parameter of the same type, but got '"
                         << type.getSuperType() << "'";
    }
    if (paramIsType && !hasCE) {
      return success();
    }
    if (paramIsType && hasCE) {
      return emitError() << "parameter '" << paramType
                         << "' of type 'Type' expects a parameter of the same type, but got '"
                         << type.getSuperType() << "'";
    }
  }
  // If it comes from a LiteralOp then the type of the generic param must be Val
  if (mlir::isa<LiteralOp>(op)) {
    if (!paramIsVal) {
      return emitError() << "parameter '" << paramType << "' was expecting a type but got " << type;
    }
    return success();
  }

  // If it comes from a SpecializeOp or a GlobalOp the type of the generic param must be Type
  if (mlir::isa<SpecializeOp, GlobalOp>(op)) {
    if (!paramIsType) {
      return emitError() << "parameter '" << paramType
                         << "' was expecting a constant 'Val' but got '" << type << "'";
    }
    return success();
  }

  // If it comes from a ConstructOp the type of the generic param must be Val and the type of the op
  // must have a ConstExpr
  if (mlir::isa<ConstructOp>(op)) {
    if (!paramIsVal) {
      return emitError() << "parameter '" << paramType
                         << "' was expecting a type but got an expression";
    }
    if (!type.hasConstExpr()) {
      return op->emitError() << "expression is not constant";
    }
    return success();
  }
  return failure();
}

} // namespace

mlir::FailureOr<TypeBinding> SpecializeTypeRule::
    typeCheck(zirgen::Zhl::SpecializeOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }

  if (operands[0].isSpecialized()) {
    return op->emitError() << "type '" << operands[0] << "' cannot be specialized";
  }

  assert(op.getArgs().size() == operands.size() - 1);

  auto genericParams = operands[0].getDeclaredGenericParams();
  if (op.getArgs().size() != genericParams.size()) {
    return op->emitError() << "type '" << operands[0] << "' expects " << genericParams.size()
                           << " generic parameters but got " << op.getArgs().size();
  }

  bool failed = false;
  for (auto [argValue, argBinding, param] :
       llvm::zip_equal(op.getArgs(), operands.drop_front(), genericParams)) {
    failed = failed || mlir::failed(checkSpecializationArg(argValue, argBinding, param, [&]() {
      return op->emitError();
    }));
  }
  if (failed) {
    return failure();
  }

  auto typeToSpecialize = operands[0];
  return interpretateOp(op, typeToSpecialize.specialize([&]() {
    return op->emitOpError();
  }, operands.drop_front()));
}

mlir::FailureOr<TypeBinding> SubscriptTypeRule::
    typeCheck(zirgen::Zhl::SubscriptOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return mlir::failure();
  }

  return interpretateOp(op, operands[0].getArrayElement([&]() { return op->emitOpError(); }));
}

mlir::FailureOr<TypeBinding> ArrayTypeRule::
    typeCheck(zirgen::Zhl::ArrayOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.empty()) {
    return op->emitOpError() << "could not infer the type of the array";
  }

  auto get = [](const TypeBinding &t) -> const TypeBinding & {
    // if (t.isKnownConst()) {
    //   return t.getSuperType();
    // }
    return t;
  };

  auto &fst = get(operands.front());
  auto commonType =
      std::reduce(operands.drop_front().begin(), operands.end(), fst, [&](auto &lhs, auto &rhs) {
    return get(lhs).commonSupertypeWith(get(rhs));
  });

  return interpretateOp(op, getBindings().Array(commonType, operands.size()));
}

mlir::FailureOr<TypeBinding> BackTypeRule::
    typeCheck(zirgen::Zhl::BackOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.size() < 2) {
    return mlir::failure();
  }
  // TODO: Check that distance is a subtype of Val
  return interpretateOp(op, operands[1]);
}

mlir::FailureOr<TypeBinding> RangeTypeRule::
    typeCheck(zirgen::Zhl::RangeOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.size() < 2) {
    return mlir::failure();
  }

  if (failed(operands[0].subtypeOf(getBindings().Get("Val")))) {
    return op->emitError()
           << "expected left side of range to be 'Val' or subtype of 'Val', but got '"
           << operands[0].getName() << "'";
  }
  if (failed(operands[1].subtypeOf(getBindings().Get("Val")))) {
    return op->emitError()
           << "expected right side of range to be 'Val' or subtype of 'Val', but got '"
           << operands[1].getName() << "'";
  }
  auto common = operands[0].commonSupertypeWith(operands[1]);

  if (operands[0].isKnownConst() && operands[1].isKnownConst()) {
    if (operands[1].getConst() < operands[0].getConst()) {
      return op->emitError() << "right side of range must be greater or equal than the left side";
    }
    return interpretateOp(
        op, getBindings().Array(common, operands[1].getConst() - operands[0].getConst())
    );
  }
  return interpretateOp(op, getBindings().UnkArray(common));
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

  // If the init value is a constant then return its super type (Val)
  if (operands[1].isConst()) {
    return interpretateOp(op, operands[1].getSuperType());
  }
  return interpretateOp(op, operands[1]);
}

mlir::FailureOr<Frame> ReduceTypeRule::allocate(Frame frame) const {
  return frame.allocateSlot<ArrayFrame>(getBindings())->getFrame();
}

mlir::FailureOr<TypeBinding> ConstructGlobalTypeRule::
    typeCheck(zirgen::Zhl::ConstructGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
        const {
  if (operands.size() < 2) {
    return mlir::failure();
  }

  return interpretateOp(op, operands[1]);
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
  super = interpretateOp(op, super);
  auto binding = TypeBinding::WithoutClosure(*super);
  binding.markSlot(nullptr);
  return binding;
}

FailureOr<Frame> BlockTypeRule::allocate(Frame frame) const {
  return frame.allocateSlot<InnerFrame>(getBindings())->getFrame();
}

FailureOr<TypeBinding> SwitchTypeRule::typeCheck(
    SwitchOp op, ArrayRef<TypeBinding> operands, Scope &scope, ArrayRef<const Scope *> regionScopes
) const {
  if (regionScopes.empty()) {
    return failure();
  }
  // TODO: Allocate a frame for the switch arms
  return interpretateOp(
      op,
      std::transform_reduce(
          regionScopes.begin(), regionScopes.end(), FailureOr<TypeBinding>(getBindings().Bottom()),
          [](FailureOr<TypeBinding> a, FailureOr<TypeBinding> b) -> FailureOr<TypeBinding> {
    if (failed(a) || failed(b)) {
      return failure();
    }
    return a->commonSupertypeWith(*b);
  }, [](const Scope *armScope) { return armScope->getSuperType(); }
      )
  );
}

FailureOr<Frame> SwitchTypeRule::allocate(Frame) const {
  return failure(); // TODO(LLZK-168)
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
  auto arrayLen = operands[0].getArraySize([&] { return op->emitError(); });
  if (failed(arrayLen)) {
    return failure();
  }

  assert(arrayLen->isConst());

  assert(!regionScopes.empty());
  auto super = regionScopes[0]->getSuperType();
  if (failed(super)) {
    return op->emitOpError() << "failed to deduce the super type";
  }

  auto binding = interpretateOp(op, getBindings().Array(*super, *arrayLen, op.getLoc()));
  if (auto slot = scope.getCurrentFrame().getParentSlot()) {
    if (auto compSlot = dyn_cast<ComponentSlot>(slot)) {
      LLVM_DEBUG(
          llvm::dbgs() << "Setting binding  of slot " << compSlot << " to " << binding << "\n"
      );
      // Is a component slot so we change the type
      compSlot->setBinding(binding);
    } else {
      LLVM_DEBUG(
          llvm::dbgs() << "Marking binding " << binding << " with slot " << slot
                       << " (name = " << slot->getSlotName() << ")\n"
      );
      // Any other slot simply gets forwarded
      binding.markSlot(slot);
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "MapOp did not mark a slot for the binding " << binding << "\n");
  }
  return binding;
}

FailureOr<Frame> MapTypeRule::allocate(Frame frame) const {
  return frame.allocateSlot<ArrayFrame>(getBindings())->getFrame();
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

  return interpretateOp(op, comp.getMember(op.getMember(), [&]() { return op->emitError(); }));
}

FailureOr<TypeBinding> DirectiveTypeRule::
    typeCheck(DirectiveOp op, ArrayRef<TypeBinding>, Scope &, ArrayRef<const Scope *>) const {
  return interpretateOp(op, getBindings().Component());
}

} // namespace zhl
