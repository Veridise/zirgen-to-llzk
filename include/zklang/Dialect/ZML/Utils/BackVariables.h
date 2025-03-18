#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/LanguageSupport/ZIR/BackVariables.h>

namespace zml {

class ZML_BVTypeProvider : public lang::zir::BVTypeProvider {
public:
  mlir::Type getMemType(mlir::Type inner) const override {
    return ComponentType::Array(inner.getContext(), inner, lang::zir::BVConstants::CYCLES);
  }

  mlir::Type getCycleType(mlir::MLIRContext *ctx) const override {
    return mlir::IndexType::get(ctx);
  }

  mlir::Type getFieldType(mlir::FlatSymbolRefAttr name, mlir::Operation *op) const override;
};

} // namespace zml
