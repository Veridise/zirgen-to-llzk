#include <llvm/ADT/StringSet.h>
#include <zklang/Dialect/ZHL/Typing/Interpreter.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

using namespace zirgen::Zhl;
using namespace mlir;

namespace zhl::expr::interpreter {

template <> TypeBinding interpretateOp(zirgen::Zhl::TypeParamOp op, const TypeBinding &binding) {
  return TypeBinding::WithExpr(binding, ConstExpr());
}

template <> TypeBinding interpretateOp(zirgen::Zhl::LiteralOp op, const TypeBinding &binding) {
  // The LiteralOp typing rule must have already created a const. Here we just check that that is
  // indeed the case.
  assert(binding.isKnownConst() && binding.getConst() == op.getValue());
  return binding;
}

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

inline ConstExpr BinaryOp(StringRef name, ArrayRef<TypeBinding> args) {
  return ConstExpr::Ctor(name, {args[0].getConstExpr(), args[1].getConstExpr()});
}

inline ConstExpr UnaryOp(StringRef name, ArrayRef<TypeBinding> args) {
  return ConstExpr::Ctor(name, {args[0].getConstExpr()});
}

} // namespace

/// Examines the construct op and checks if it could be used to construct an affine expression. If
/// the constructed component does not match with an semi-affine operator or any of the operands do
/// not have a ConstExpr then gives up an returns the binding without a ConstExpr. A binding that
/// does not have a ConstExpr means that it does not encode a semi-affine expression.
template <>
TypeBinding
interpretateOp(zirgen::Zhl::ConstructOp, const TypeBinding &binding, ArrayRef<TypeBinding> &&args) {
  if (!AffineRepresentableComponents.contains(binding.getName())) {
    return TypeBinding::NoExpr(binding);
  }
  if (binding.getName() == "Neg") {
    assert(args.size() == 1);
  } else {
    assert(args.size() == 2);
  }

  auto tok = getToken(binding.getName());
  assert(succeeded(tok));
  switch (*tok) {
  case Cmps::Add:
    // semi-affine-expr `+` semi-affine-expr
    if (isSemiAffine(args[0]) && isSemiAffine(args[1])) {
      return TypeBinding::WithExpr(binding, BinaryOp("Add", args));
    }
    break;
  case Cmps::Sub:
    // semi-affine-expr `-` semi-affine-expr
    if (isSemiAffine(args[0]) && isSemiAffine(args[1])) {
      return TypeBinding::WithExpr(binding, BinaryOp("Sub", args));
    }
    break;
  case Cmps::Mul:
    // symbol-or-const `*` semi-affine-expr
    if (isSymbolOrConst(args[0]) && isSemiAffine(args[1])) {
      return TypeBinding::WithExpr(binding, BinaryOp("Mul", args));
    }
    break;
  case Cmps::Div:
    // semi-affine-expr `floordiv` symbol-or-const
    if (isSemiAffine(args[0]) && isSymbolOrConst(args[1])) {
      return TypeBinding::WithExpr(binding, BinaryOp("Div", args));
    }
    break;
  case Cmps::Mod:
    // semi-affine-expr `mod` symbol-or-const
    if (isSemiAffine(args[0]) && isSymbolOrConst(args[1])) {
      return TypeBinding::WithExpr(binding, BinaryOp("Mod", args));
    }
    break;
  case Cmps::Neg:
    // `-` integer-literal
    if (isIntLit(args[0])) {
      return TypeBinding::WithExpr(binding, UnaryOp("Neg", args));
    }
    break;
  }

  return TypeBinding::NoExpr(binding);
}

} // namespace zhl::expr::interpreter
