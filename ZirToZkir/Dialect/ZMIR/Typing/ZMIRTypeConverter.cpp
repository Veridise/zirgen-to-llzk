#include "ZMIRTypeConverter.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <mlir/IR/BuiltinOps.h>
#include <optional>

using namespace zkc::Zmir;
using namespace zkc;
using namespace zirgen;

mlir::Value findTypeInUseDefChain(mlir::Value v,
                                  const mlir::TypeConverter *converter) {
  mlir::Value cur = v;
  while (cur.getDefiningOp() &&
         mlir::isa<mlir::UnrealizedConversionCastOp>(cur.getDefiningOp())) {
    auto ops = cur.getDefiningOp()->getOperands();
    if (ops.size() != 1) {
      return v; // Give up if multiple operands
    }
    cur = ops[0];
  }

  if (Zmir::isValidZmirType(cur.getType()) &&
      converter->isLegal(cur.getType())) {
    return cur;
  }
  return v;
}

void findTypesInUseDefChain(mlir::ValueRange r,
                            const mlir::TypeConverter *converter,
                            llvm::SmallVector<mlir::Value> &results) {
  std::transform(
      r.begin(), r.end(), std::back_inserter(results),
      [&](mlir::Value v) { return findTypeInUseDefChain(v, converter); });
}

ZMIRTypeConverter::ZMIRTypeConverter() {
  addConversion([](mlir::Type t) { return t; });
  addConversion(
      [](Zhl::ExprType t) { return Zmir::PendingType::get(t.getContext()); });

  addSourceMaterialization(
      [](mlir::OpBuilder &builder, Zhl::ExprType type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
        return builder
            .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
            .getResult(0);
      });
  addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;

        // Try to find a previous value in the use-def chain that is valid.
        auto value = findTypeInUseDefChain(inputs[0], type);
        if (mlir::succeeded(value))
          return *value;
        // Otherwise materialize a conversion
        return builder
            .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
            .getResult(0);
      });
  addTargetMaterialization(
      [](mlir::OpBuilder &builder, Zmir::ValType type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
        if (auto compTyp =
                mlir::dyn_cast<Zmir::ComponentType>(inputs[0].getType())) {
          if (compTyp.getName().getValue() != "Component")
            return builder.create<Zmir::ReadSuperTransOp>(
                loc, Zmir::ValType::get(builder.getContext()), inputs[0]);
        }
        return std::nullopt;
      });

  addArgumentMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
        return builder
            .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
            .getResult(0);
      });
  addArgumentMaterialization(
      [](mlir::OpBuilder &builder, Zmir::ValType type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
        if (auto compTyp =
                mlir::dyn_cast<Zmir::ComponentType>(inputs[0].getType())) {
          if (compTyp.getName().getValue() != "Component")
            return builder.create<Zmir::ReadSuperTransOp>(
                loc, Zmir::ValType::get(builder.getContext()), inputs[0]);
        }

        return std::nullopt;
      });
}

/// Climbs the use def chain until it finds a
/// value that is not defined by an unrealized cast
/// and that is of the target type. Returns that type.
/// If that doesn't happen returns the initial type.
mlir::FailureOr<mlir::Value>
ZMIRTypeConverter::findTypeInUseDefChain(mlir::Value v, mlir::Type targetType) {
  mlir::Value cur = v;
  std::vector<mlir::Value> acc;
  while (cur.getDefiningOp() &&
         mlir::isa<mlir::UnrealizedConversionCastOp>(cur.getDefiningOp())) {
    auto ops = cur.getDefiningOp()->getOperands();
    if (ops.size() != 1) {
      return mlir::failure(); // Give up if multiple operands
    }
    cur = ops[0];
    // Accumulate the valid values
    if (cur.getType() == targetType) {
      acc.push_back(cur);
    }
  }

  // Returns the highest value to propiciate as must dead-code as possible for
  // later.
  if (!acc.empty())
    return acc.back();
  /*if (cur.getType() == targetType) {*/
  /*  return cur;*/
  /*}*/
  return mlir::failure();
}
