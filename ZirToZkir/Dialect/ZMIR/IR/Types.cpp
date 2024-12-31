#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <algorithm>
#include <iterator>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

#include "ZirToZkir/Dialect/ZMIR/IR/TypeInterfaces.inc.cpp"

namespace zkc::Zmir {

bool isValidZmirType(mlir::Type type) {
  return llvm::isa<TypeVarType>(type) || llvm::isa<StringType>(type) ||
         llvm::isa<UnionType>(type) || llvm::isa<ValType>(type) || llvm::isa<ComponentType>(type) ||
         llvm::isa<PendingType>(type) ||
         (llvm::isa<VarArgsType>(type) && isValidZmirType(llvm::cast<VarArgsType>(type).getInner())
         ) ||
         (llvm::isa<ArrayType>(type) && isValidZmirType(llvm::cast<ArrayType>(type).getInnerType())
         );
}

inline mlir::LogicalResult
checkValidZmirType(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type) {
  if (!isValidZmirType(type)) {
    return emitError() << "expected " << "a valid ZMIR type" << " but found " << type;
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
    ComponentType superType, ::llvm::ArrayRef<::mlir::Attribute> params
) {
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

mlir::LogicalResult BoundedArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type elementType,
    mlir::Attribute size
) {
  auto typeRes = checkValidZmirType(emitError, elementType);
  auto sizeRes = checkValidParam(emitError, size);

  return mlir::success(mlir::succeeded(typeRes) && mlir::succeeded(sizeRes));
}

int64_t BoundedArrayType::getSizeInt() {
  if (llvm::isa<mlir::IntegerAttr>(getSize())) {
    mlir::IntegerAttr i = llvm::cast<mlir::IntegerAttr>(getSize());
    return i.getValue().getZExtValue();
  }
  return 0;
}

mlir::LogicalResult UnboundedArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type
) {
  return checkValidZmirType(emitError, type);
}

mlir::Attribute UnboundedArrayType::getSize() {
  return mlir::IntegerAttr::get(
      mlir::IntegerType::get(getContext(), 64, mlir::IntegerType::Signed), getSizeInt()
  );
}

} // namespace zkc::Zmir
