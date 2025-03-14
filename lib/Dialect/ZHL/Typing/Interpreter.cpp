#include <zklang/Dialect/ZHL/Typing/Interpreter.h>

#include <algorithm>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

#define DEBUG_TYPE "zhl-interpreter"

using namespace zirgen::Zhl;
using namespace mlir;

namespace zhl::expr {

namespace {

/********************************************************************************************
 *
 * The grammar to represent semi-affine expressions with MLIR is as follows:
 *
 * semi-affine-expr ::= `(` semi-affine-expr `)`
 *                | semi-affine-expr `+` semi-affine-expr
 *                | semi-affine-expr `-` semi-affine-expr
 *                | symbol-or-const `*` semi-affine-expr
 *                | semi-affine-expr `ceildiv` symbol-or-const
 *                | semi-affine-expr `floordiv` symbol-or-const
 *                | semi-affine-expr `mod` symbol-or-const
 *                | bare-id
 *                | `-`? integer-literal
 *
 * symbol-or-const ::= `-`? integer-literal | symbol-id
 *
 * The interpreter function for TypeParamOp and LiteralOp are affine by definition, and the only
 * place where we need to check if the expression is going to be affine is in the interpreter
 * function for ConstructOp. Each component in the set below corresponds to an operation in the
 * grammar above, with `Div` as a special case since it can map to two different division
 * operations. In this implementation the `Div` component represents the `floordiv` affine operator.
 *
 * The definition of semi-affine is not completely correct with respect to MLIR's, a single
 * symbol-id will be considered a semi-affine expression. However, when materializing the expression
 * into an attribute a ConstExpr of type SymExpr will generate a FlatSymbolRefAttr and not an
 * AffineMapAttr, which removes the corner case from the actual semi-affine expressions when
 * generated.
 *
 ******************************************************************************************/

// Set of components that could potentially be used to construct an affine expression
static llvm::StringSet<> AffineRepresentableComponents({"Add", "Sub", "Mul", "Div", "Mod", "Neg"});

inline bool isSemiAffine(const TypeBinding &binding) { return binding.hasConstExpr(); }

/// A binding is a symbol or constant if its ConstExpr is either a ValExpr or a SymExpr, or if its
/// the result of constructing a Neg.
inline bool isSymbolOrConst(const TypeBinding &binding) {
  return isSemiAffine(binding) &&
         (mlir::isa<expr::ValExpr, expr::SymExpr>(binding.getConstExpr()) ||
          mlir::cast<expr::CtorExpr>(binding.getConstExpr()).getTypeName() == "Neg");
}

inline bool isIntLit(const TypeBinding &binding) {
  return isSemiAffine(binding) && mlir::isa<expr::ValExpr>(binding.getConstExpr());
}

enum class Cmps : uint8_t { Add = 0, Sub, Mul, Div, Mod, Neg };

FailureOr<Cmps> getToken(StringRef name) {
  return llvm::StringSwitch<FailureOr<Cmps>>(name)
      .Case("Add", Cmps::Add)
      .Case("Sub", Cmps::Sub)
      .Case("Mul", Cmps::Mul)
      .Case("Div", Cmps::Div)
      .Case("Mod", Cmps::Mod)
      .Case("Neg", Cmps::Neg)
      .Default(failure());
}

inline uint8_t arity(Cmps cmp) {
  switch (cmp) {
  case Cmps::Add:
  case Cmps::Sub:
  case Cmps::Mul:
  case Cmps::Div:
  case Cmps::Mod:
    return 2;
  case Cmps::Neg:
    return 1;
  }
  llvm_unreachable("switch case with closed values");
  return 0;
}

inline ConstExpr BinaryOp(StringRef name, ArrayRef<TypeBinding> args) {
  return ConstExpr::Ctor(name, {args[0].getConstExpr(), args[1].getConstExpr()});
}

inline ConstExpr UnaryOp(StringRef name, ArrayRef<TypeBinding> args) {
  return ConstExpr::Ctor(name, {args[0].getConstExpr()});
}

} // namespace

namespace interpreter {

template <> TypeBinding interpretateOp(zirgen::Zhl::TypeParamOp op, const TypeBinding &binding) {
  if (binding.getSuperType().isVal()) {
    return TypeBinding::WithExpr(binding, ConstExpr::Symbol(op.getName(), op.getIndex()));
  }
  return TypeBinding::NoExpr(binding);
}

template <> TypeBinding interpretateOp(zirgen::Zhl::LiteralOp op, const TypeBinding &binding) {
  // The LiteralOp typing rule must have already created a const. Here we just check that that is
  // indeed the case.
  assert(binding.isKnownConst() && binding.getConst() == op.getValue());
  return binding;
}

/// Examines the construct op and checks if it could be used to construct an affine expression. If
/// the constructed component does not match with an semi-affine operator or any of the operands do
/// not have a ConstExpr then gives up an returns the binding without a ConstExpr. A binding that
/// does not have a ConstExpr means that it does not encode a semi-affine expression.
template <>
TypeBinding interpretateOp(
    zirgen::Zhl::ConstructOp op, const TypeBinding &binding, ArrayRef<TypeBinding> &&args
) {
  LLVM_DEBUG(
      llvm::dbgs() << "interpretating " << op << " at " << op.getLoc() << " with type " << binding
                   << "\n"
  );
  if (!AffineRepresentableComponents.contains(binding.getName())) {
    LLVM_DEBUG(llvm::dbgs() << "  is not affine representable\n");
    return TypeBinding::NoExpr(binding);
  }

  auto tok = getToken(binding.getName());
  assert(succeeded(tok));
  assert(args.size() == arity(*tok));
  switch (*tok) {
  case Cmps::Add:
    // semi-affine-expr `+` semi-affine-expr
    if (isSemiAffine(args[0]) && isSemiAffine(args[1])) {
      LLVM_DEBUG(llvm::dbgs() << "  valid affine Add\n");
      return TypeBinding::WithExpr(binding, BinaryOp("Add", args));
    }
    LLVM_DEBUG(llvm::dbgs() << "  invalid affine Add\n");
    break;
  case Cmps::Sub:
    // semi-affine-expr `-` semi-affine-expr
    if (isSemiAffine(args[0]) && isSemiAffine(args[1])) {
      LLVM_DEBUG(llvm::dbgs() << "  valid affine Sub\n");
      return TypeBinding::WithExpr(binding, BinaryOp("Sub", args));
    }
    LLVM_DEBUG(llvm::dbgs() << "  invalid affine Sub\n");
    break;
  case Cmps::Mul:
    // symbol-or-const `*` semi-affine-expr
    if (isSymbolOrConst(args[0]) && isSemiAffine(args[1])) {
      LLVM_DEBUG(llvm::dbgs() << "  valid affine Mul\n");
      return TypeBinding::WithExpr(binding, BinaryOp("Mul", args));
    }
    LLVM_DEBUG(llvm::dbgs() << "  invalid affine Mul\n");
    break;
  case Cmps::Div:
    // semi-affine-expr `floordiv` symbol-or-const
    if (isSemiAffine(args[0]) && isSymbolOrConst(args[1])) {
      LLVM_DEBUG(llvm::dbgs() << "  valid affine Div\n");
      return TypeBinding::WithExpr(binding, BinaryOp("Div", args));
    }
    LLVM_DEBUG(
        llvm::dbgs() << "  invalid affine Div | lhs = " << isSemiAffine(args[0])
                     << ", rhs = " << isSymbolOrConst(args[1]) << "\n"
    );
    break;
  case Cmps::Mod:
    // semi-affine-expr `mod` symbol-or-const
    if (isSemiAffine(args[0]) && isSymbolOrConst(args[1])) {
      LLVM_DEBUG(llvm::dbgs() << "  valid affine Mod\n");
      return TypeBinding::WithExpr(binding, BinaryOp("Mod", args));
    }
    LLVM_DEBUG(llvm::dbgs() << "  invalid affine Mod\n");
    break;
  case Cmps::Neg:
    // `-` integer-literal
    if (isIntLit(args[0])) {
      LLVM_DEBUG(llvm::dbgs() << "  valid affine Neg\n");
      return TypeBinding::WithExpr(binding, UnaryOp("Neg", args));
    }
    // XXX: May have to do tricks for negations of non literal integers like -N => 0 - N
    LLVM_DEBUG(llvm::dbgs() << "  invalid affine Neg\n");
    break;
  }

  return TypeBinding::NoExpr(binding);
}

} // namespace interpreter

namespace detail {

// Defined here to access the semi-affine support information

FailureOr<AffineExpr> Val::convertIntoAffineExpr(Builder &builder) const {
  return builder.getAffineConstantExpr(value);
}

FailureOr<AffineExpr> Symbol::convertIntoAffineExpr(Builder &builder) const {
  return builder.getAffineSymbolExpr(pos);
}

FailureOr<AffineExpr> Ctor::convertIntoAffineExpr(Builder &builder) const {
  // If the component this expression builds is not valid return failure
  auto tok = getToken(typeName);
  if (failed(tok)) {
    return failure();
  }
  assert(args.size() == arity(*tok));

  SmallVector<AffineExpr, 2> exprs;
  for (auto &arg : args) {
    auto expr = arg.convertIntoAffineExpr(builder);
    if (failed(expr)) {
      return failure();
    }
    exprs.push_back(*expr);
  }

  switch (*tok) {
  case Cmps::Add:
    return exprs[0] + exprs[1];
  case Cmps::Sub:
    return exprs[0] - exprs[1];
  case Cmps::Mul:
    return exprs[0] * exprs[1];
  case Cmps::Div:
    return exprs[0].floorDiv(exprs[1]);
  case Cmps::Mod:
    return exprs[0] % exprs[1];
  case Cmps::Neg:
    return -exprs[0];
  }

  return failure();
}

} // namespace detail

} // namespace zhl::expr
