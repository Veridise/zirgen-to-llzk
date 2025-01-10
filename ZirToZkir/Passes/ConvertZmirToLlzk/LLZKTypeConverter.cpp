
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

mlir::Type deduceArrayType(mlir::Attribute attr) {
  if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
    return typeAttr.getValue();
  }
  llvm::dbgs() << "attr = " << attr << "\n";
  assert(false && "Failed to convert array type");
  return nullptr;
}

bool arrayLenIsKnown(mlir::Attribute attr) { return mlir::isa<mlir::IntegerAttr>(attr); }

int64_t getSize(mlir::Attribute attr) {
  auto intAttr = mlir::cast<mlir::IntegerAttr>(attr);
  return intAttr.getValue().getZExtValue();
}

void convertParamAttrs(
    mlir::ArrayRef<mlir::Attribute> in, llvm::SmallVector<mlir::Attribute> &out,
    llzk::LLZKTypeConverter &converter
) {
  std::transform(
      in.begin(), in.end(), std::back_inserter(out),
      [&](mlir::Attribute attr) -> mlir::Attribute {
    if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
      return mlir::TypeAttr::get(converter.convertType(typeAttr.getValue()));
    }
    return attr;
  }
  );
}

mlir::SymbolRefAttr getSizeSym(mlir::Attribute attr) {
  auto sym = mlir::dyn_cast<mlir::SymbolRefAttr>(attr);
  if (!sym) {
    llvm::dbgs() << "attr = " << attr << "\n";
  }
  assert(sym && "was expecting a symbol");
  return sym;
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
    if (t.getName().getValue() == "Array") {
      assert(t.getParams().size() == 2);
      auto typeAttr = t.getParams()[0];
      auto sizeAttr = t.getParams()[1];
      if (arrayLenIsKnown(sizeAttr)) {
        return llzk::ArrayType::get(convertType(deduceArrayType(typeAttr)), {getSize(sizeAttr)});
      } else {
        return llzk::ArrayType::get(convertType(deduceArrayType(typeAttr)), {getSizeSym(sizeAttr)});
      }
    }
    llvm::SmallVector<mlir::Attribute> convertedAttrs;
    convertParamAttrs(t.getParams(), convertedAttrs, *this);
    return llzk::StructType::get(
        t.getContext(), t.getName(), mlir::ArrayAttr::get(t.getContext(), convertedAttrs)
    );
  });

  addConversion([&](Zmir::VarArgsType t) {
    std::vector<int64_t> shape = {mlir::ShapedType::kDynamic};
    return llzk::ArrayType::get(convertType(t.getInner()), shape);
  });

  addConversion([](Zmir::TypeVarType t) {
    return llzk::TypeVarType::get(t.getContext(), t.getName());
  });

  addSourceMaterialization(unrealizedCastMaterialization);
  addTargetMaterialization(unrealizedCastMaterialization);
  addArgumentMaterialization(unrealizedCastMaterialization);
}
