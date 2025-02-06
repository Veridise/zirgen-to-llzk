
#include "zklang/Passes/ConvertZmlToLlzk/LLZKTypeConverter.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include <algorithm>
#include <iterator>
#include <llvm/Support/FileSystem.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>
#include <optional>

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

llzk::LLZKTypeConverter::LLZKTypeConverter(mlir::Operation *stRoot)
    : feltEquivalentTypes({"Val", "Add", "Sub", "Mul", "BitAnd", "Inv", "Isz"}),
      symbolTableRoot(stRoot) {

  addConversion([](mlir::Type t) { return t; });

  // Conversions from ZML to LLZK

  addConversion([&](Zmir::ComponentType t) -> mlir::Type {
    // if (feltEquivalentTypes.find(t.getName().getValue()) != feltEquivalentTypes.end() &&
    //     t.getBuiltin()) {
    //   return llzk::FeltType::get(t.getContext());
    // }
    // if (t.getName().getValue() == "String") {
    //   return llzk::StringType::get(t.getContext());
    // }
    // if (t.getName().getValue() == "Array") {
    //   assert(t.getParams().size() == 2);
    //   auto typeAttr = t.getParams()[0];
    //   auto sizeAttr = t.getParams()[1];
    //   // TODO Group together arrays of arrays
    //   if (arrayLenIsKnown(sizeAttr)) {
    //     return llzk::ArrayType::get(convertType(deduceArrayType(typeAttr)), {getSize(sizeAttr)});
    //   } else {
    //     return llzk::ArrayType::get(convertType(deduceArrayType(typeAttr)),
    //     {getSizeSym(sizeAttr)});
    //   }
    // }
    llvm::SmallVector<mlir::Attribute> convertedAttrs;
    convertParamAttrs(t.getParams(), convertedAttrs, *this);
    return llzk::StructType::get(
        t.getContext(), t.getName(), mlir::ArrayAttr::get(t.getContext(), convertedAttrs)
    );
  });

  addConversion([&](Zmir::ComponentType t) -> std::optional<mlir::Type> {
    if (t.getName().getValue() != "Array") {
      return std::nullopt;
    }
    assert(t.getParams().size() == 2);
    auto typeAttr = t.getParams()[0];
    auto sizeAttr = t.getParams()[1];
    // TODO Group together arrays of arrays
    if (arrayLenIsKnown(sizeAttr)) {
      return llzk::ArrayType::get(convertType(deduceArrayType(typeAttr)), {getSize(sizeAttr)});
    } else {
      return llzk::ArrayType::get(convertType(deduceArrayType(typeAttr)), {getSizeSym(sizeAttr)});
    }
  });

  addConversion([](Zmir::ComponentType t) -> std::optional<mlir::Type> {
    if (t.getName().getValue() == "String") {
      return llzk::StringType::get(t.getContext());
    }
    return std::nullopt;
  });

  addConversion([&](Zmir::ComponentType t) -> std::optional<mlir::Type> {
    if (feltEquivalentTypes.find(t.getName().getValue()) != feltEquivalentTypes.end() &&
        t.getBuiltin()) {
      return llzk::FeltType::get(t.getContext());
    }
    return std::nullopt;
  });

  addConversion([&](Zmir::VarArgsType t) {
    std::vector<int64_t> shape = {mlir::ShapedType::kDynamic};
    return llzk::ArrayType::get(convertType(t.getInner()), shape);
  });

  addConversion([](Zmir::TypeVarType t) {
    return llzk::TypeVarType::get(t.getContext(), t.getName());
  });

  // XXX: This may not be necessary
  // Conversions from LLZK to ZML

  // addConversion([&](llzk::StructType t) -> std::optional<mlir::Type> {
  //   auto lookupResult = t.getDefinition(stc, symbolTableRoot);
  //   if (mlir::failed(lookupResult)) {
  //     return std::nullopt;
  //   }
  //   auto fieldDef =
  //       lookupResult->get().getFieldDef(mlir::StringAttr::get(t.getContext(), "$super"));
  //   if (!fieldDef) {
  //     return std::nullopt;
  //   }
  //   auto superType = convertType(fieldDef.getType());
  //   return Zmir::ComponentType::get(
  //       t.getContext(), t.getNameRef().getLeafReference(), superType, t.getParams(), false
  //   );
  // });

  // TODO llzk::FeltType -> Zmir::ComponentType
  // TODO llzk::StringType -> Zmir::ComponentType
  // TODO llzk::ArrayType -> Zmir::ComponentType
  // TODO llzk::TypeVarType -> Zmir.TypeVarType

  addSourceMaterialization(unrealizedCastMaterialization);
  addTargetMaterialization(unrealizedCastMaterialization);
  addArgumentMaterialization(unrealizedCastMaterialization);
}
