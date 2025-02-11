#include "zklang/Dialect/ZML/IR/Types.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include <algorithm>
#include <iterator>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

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
  if (llvm::isa<mlir::SymbolRefAttr, mlir::IntegerAttr, mlir::TypeAttr>(attr)) {
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
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, ::mlir::FlatSymbolRefAttr name,
    mlir::Type superType, ::llvm::ArrayRef<::mlir::Attribute> params, bool builtin
) {
  if (!superType && name.getValue() != "Component") {
    return emitError() << "malformed IR: super type for " << name << " cannot be null";
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
  auto comp = symbolTable.lookupNearestSymbolFrom(op, getName());
  if (!comp) {
    return nullptr;
  }

  return mlir::dyn_cast<ComponentInterface>(comp);
}

} // namespace zml
