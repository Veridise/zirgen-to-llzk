#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <algorithm>
#include <iterator>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

namespace zkc::Zmir {

bool isValidZmirType(mlir::Type type) {
  return llvm::isa<TypeVarType>(type) || llvm::isa<StringType>(type) ||
         llvm::isa<UnionType>(type) || llvm::isa<ValType>(type) ||
         llvm::isa<ComponentType>(type) ||
         /*llvm::isa<zirgen::Zhl::ExprType>(type) ||*/
         llvm::isa<PendingType>(type) ||
         (llvm::isa<ArrayType>(type) &&
          isValidZmirType(llvm::cast<ArrayType>(type).getInnerType()));
}

inline mlir::LogicalResult
checkValidZmirType(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   mlir::Type type) {
  if (!isValidZmirType(type)) {
    return emitError() << "expected " << "a valid ZMIR type" << " but found "
                       << type;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult
checkValidConstParam(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                     mlir::Attribute attr) {
  if (!(llvm::isa<mlir::SymbolRefAttr>(attr) ||
        llvm::isa<mlir::IntegerAttr>(attr))) {
    return emitError()
           << "expected either a symbol or a literal integer but got " << attr;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult
checkValidTypeParam(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    mlir::Attribute attr) {
  // TODO: Check for types passed as attributes.
  if (!llvm::isa<mlir::SymbolRefAttr>(attr)) {
    return emitError() << "expected a symbol but got " << attr;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult
ComponentType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      ::mlir::FlatSymbolRefAttr name,
                      ::llvm::ArrayRef<::mlir::Attribute> typeParams,
                      ::llvm::ArrayRef<::mlir::Attribute> constParams) {
  std::vector<mlir::LogicalResult> results;
  std::transform(typeParams.begin(), typeParams.end(),
                 std::back_inserter(results), [&](mlir::Attribute attr) {
                   return checkValidTypeParam(emitError, attr);
                 });
  std::transform(constParams.begin(), constParams.end(),
                 std::back_inserter(results), [&](mlir::Attribute attr) {
                   return checkValidConstParam(emitError, attr);
                 });

  if (results.empty())
    return mlir::success();
  return mlir::success(
      std::all_of(results.begin(), results.end(), mlir::succeeded));
}

mlir::LogicalResult
ArrayType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  mlir::Type elementType, mlir::Attribute size) {
  auto typeRes = checkValidZmirType(emitError, elementType);
  auto sizeRes = checkValidConstParam(emitError, size);

  return mlir::success(mlir::succeeded(typeRes) && mlir::succeeded(sizeRes));
}

uint64_t ArrayType::getSizeInt() {
  if (llvm::isa<mlir::IntegerAttr>(getSize())) {
    mlir::IntegerAttr i = llvm::cast<mlir::IntegerAttr>(getSize());
    return i.getValue().getZExtValue();
  }
  return 0;
}

} // namespace zkc::Zmir
