#include <zklang/Dialect/ZML/IR/Types.h>

#include <algorithm>
#include <iterator>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

using namespace mlir;

namespace zml {

bool isValidZMLType(mlir::Type type) {
  return llvm::isa<TypeVarType>(type) || llvm::isa<ComponentType>(type) ||
         (llvm::isa<VarArgsType>(type) && isValidZMLType(llvm::cast<VarArgsType>(type).getInner()));
}

inline mlir::LogicalResult
checkValidZmirType(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type) {
  if (!isValidZMLType(type)) {
    return emitError() << "expected " << "a valid ZML type" << " but found " << type;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult
checkValidParam(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Attribute attr) {
  if (llvm::isa<mlir::SymbolRefAttr, mlir::IntegerAttr, mlir::TypeAttr, ConstExprAttr>(attr)) {
    return mlir::success();
  }
  return emitError() << "expected either a symbol or a literal integer but got " << attr;
}

mlir::LogicalResult checkValidTypeParam(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Attribute attr
) {
  // TODO: Check for types passed as attributes.
  if (!llvm::isa<mlir::SymbolRefAttr>(attr)) {
    return emitError() << "expected a symbol but got " << attr;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult ComponentType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, ::mlir::FlatSymbolRefAttr compName,
    mlir::Type superType, ::llvm::ArrayRef<::mlir::Attribute> params, bool builtin
) {
  if (!superType && compName.getValue() != "Component") {
    return emitError() << "malformed IR: super type for " << compName << " cannot be null";
  }
  // TODO: Maybe add a check that ensures that only known builtins can have the flag set to true?
  if (superType) {
    if (!mlir::isa<ComponentType, TypeVarType>(superType)) {
      return emitError() << "unexpected type " << superType;
    }
  }
  std::vector<mlir::LogicalResult> results;
  std::transform(
      params.begin(), params.end(), std::back_inserter(results),
      [&](mlir::Attribute attr) { return checkValidParam(emitError, attr); }
  );

  if (results.empty()) {
    return mlir::success();
  }
  return mlir::success(std::all_of(results.begin(), results.end(), mlir::succeeded));
}

ComponentInterface
ComponentType::getDefinition(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op) {
  return mlir::dyn_cast_if_present<ComponentInterface>(
      symbolTable.lookupNearestSymbolFrom(op, getName())
  );
}

FailureOr<Type> ComponentType::getFirstMatchingSuperType(llvm::function_ref<bool(Type)> pred
) const {
  if (pred(*this)) {
    return *this;
  }
  if (auto super = getSuperType()) {
    if (auto superComp = mlir::dyn_cast<ComponentType>(super)) {
      return superComp.getFirstMatchingSuperType(pred);
    } else if (pred(super)) {
      return super;
    }
  }
  return failure();
}

FailureOr<Attribute> ComponentType::getArraySize() const {
  if (!isArray()) {
    return failure();
  }
  if (getName().getValue() == "Array") {
    assert(getParams().size() == 2 && "Arrays must have only two params by definition");
    return getParams()[1];
  }
  if (!getSuperTypeAsComp()) {
    return failure();
  }
  return getSuperTypeAsComp().getArraySize();
}

ComponentType ComponentType::getSuperTypeAsComp() const {
  if (auto comp = mlir::dyn_cast_if_present<ComponentType>(getSuperType())) {
    return comp;
  }
  return nullptr;
}

} // namespace zml
