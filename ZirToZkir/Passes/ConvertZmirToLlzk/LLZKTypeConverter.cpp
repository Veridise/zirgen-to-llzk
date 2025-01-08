
#include "LLZKTypeConverter.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include <mlir/IR/BuiltinOps.h>

using namespace zkc::Zmir;
using namespace zkc;

std::optional<mlir::Value> unrealizedCastMaterialization(
    mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs, mlir::Location loc
) {

  assert(inputs.size() == 1);
  return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
}

llzk::LLZKTypeConverter::LLZKTypeConverter()
    : feltEquivalentTypes({"Val", "Add", "Sub", "Mul", "BitAnd", "Inv"}) {
  addConversion([](mlir::Type t) { return t; });

  addConversion([&](Zmir::ComponentType t) -> mlir::Type {
    if (feltEquivalentTypes.find(t.getName().getValue()) != feltEquivalentTypes.end()) {
      return llzk::FeltType::get(t.getContext());
    }
    if (t.getName().getValue() == "String") {
      return llzk::StringType::get(t.getContext());
    }
    return llzk::StructType::get(
        t.getContext(), t.getName(), mlir::ArrayAttr::get(t.getContext(), t.getParams())
    );
  });

  addConversion([&](Zmir::VarArgsType t) {
    std::vector<int64_t> shape = {mlir::ShapedType::kDynamic};
    return llzk::ArrayType::get(convertType(t.getInner()), shape);
  });

  addSourceMaterialization(unrealizedCastMaterialization);
  addTargetMaterialization(unrealizedCastMaterialization);
  addArgumentMaterialization(unrealizedCastMaterialization);
}
