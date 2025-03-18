#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/LanguageSupport/ZIR/BackVariables.h>

namespace zml {

class ZML_BVDialectHelper : public lang::zir::BVDialectHelper {
public:
  mlir::Type getMemType(mlir::Type inner) const override {
    return ComponentType::Array(inner.getContext(), inner, lang::zir::BVConstants::CYCLES);
  }

  mlir::Type getCycleType(mlir::MLIRContext *ctx) const override {
    return mlir::IndexType::get(ctx);
  }

  mlir::Value getCycleConstant(mlir::OpBuilder &, uint64_t, mlir::Location) const override;

  mlir::Value allocateMem(mlir::OpBuilder &, mlir::Type, mlir::Location) const override;

  mlir::Value readArray(
      mlir::Type, mlir::Value array, mlir::Value n, mlir::OpBuilder &, mlir::Location
  ) const override;

  void writeArray(
      mlir::Value array, mlir::Value n, mlir::Value in, mlir::OpBuilder &, mlir::Location
  ) const override;

  mlir::Value readField(
      mlir::Type, mlir::FlatSymbolRefAttr name, mlir::Value src, mlir::OpBuilder &, mlir::Location
  ) const override;

  mlir::Type deduceComponentType(mlir::FunctionOpInterface) const override;
};

} // namespace zml
