
#include <algorithm>
#include <iterator>
#include <llvm/Support/FileSystem.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/Types.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>
#include <optional>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Passes/ConvertZmlToLlzk/LLZKTypeConverter.h>

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
    : feltEquivalentTypes({"Val", "Add", "Sub", "Mul", "BitAnd", "Inv", "Isz", "InRange"}) {

  addConversion([](mlir::Type t) { return t; });

  // Conversions from ZML to LLZK

  addConversion([&](zml::ComponentType t) -> mlir::Type {
    llvm::SmallVector<mlir::Attribute> convertedAttrs;
    convertParamAttrs(t.getParams(), convertedAttrs, *this);
    return llzk::StructType::get(
        t.getContext(), t.getName(), mlir::ArrayAttr::get(t.getContext(), convertedAttrs)
    );
  });

  addConversion([&](zml::ComponentType t) -> std::optional<mlir::Type> {
    if (t.getName().getValue() != "Array") {
      return std::nullopt;
    }
    assert(t.getParams().size() == 2);
    auto typeAttr = t.getParams()[0];
    auto sizeAttr = t.getParams()[1];
    // TODO(LLZK-173) Group together arrays of arrays
    if (arrayLenIsKnown(sizeAttr)) {
      return llzk::ArrayType::get(convertType(deduceArrayType(typeAttr)), {getSize(sizeAttr)});
    } else {
      return llzk::ArrayType::get(convertType(deduceArrayType(typeAttr)), {getSizeSym(sizeAttr)});
    }
  });

  addConversion([](zml::ComponentType t) -> std::optional<mlir::Type> {
    if (t.getName().getValue() == "String") {
      return llzk::StringType::get(t.getContext());
    }
    return std::nullopt;
  });

  addConversion([&](zml::ComponentType t) -> std::optional<mlir::Type> {
    if (feltEquivalentTypes.find(t.getName().getValue()) != feltEquivalentTypes.end() &&
        t.getBuiltin()) {
      return llzk::FeltType::get(t.getContext());
    }
    return std::nullopt;
  });

  addConversion([&](zml::VarArgsType t) {
    std::vector<int64_t> shape = {mlir::ShapedType::kDynamic};
    return llzk::ArrayType::get(convertType(t.getInner()), shape);
  });

  addConversion([](zml::TypeVarType t) {
    return llzk::TypeVarType::get(t.getContext(), t.getName());
  });

  addSourceMaterialization(unrealizedCastMaterialization);
  addTargetMaterialization(unrealizedCastMaterialization);
  addArgumentMaterialization(unrealizedCastMaterialization);
}
