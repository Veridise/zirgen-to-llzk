#include "ZMIRTypeConverter.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <mlir/IR/BuiltinOps.h>

using namespace zkc::Zmir;
using namespace zkc;
using namespace zirgen;

ZMIRTypeConverter::ZMIRTypeConverter() {
  addConversion([](mlir::Type t) { return t; });
  addConversion(
      [](Zhl::ExprType t) { return Zmir::PendingType::get(t.getContext()); });
  /*addConversion(*/
  /*    [](Zhl::ExprType t) { return Zmir::StringType::get(t.getContext());
   * });*/
  /*addConversion(*/
  /*    [](Zhl::ExprType t) { return Zmir::ValType::get(t.getContext()); });*/

  addSourceMaterialization(
      [](mlir::OpBuilder &builder, Zhl::ExprType type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
        return builder
            .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
            .getResult(0);
      });
  addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
        return builder
            .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
            .getResult(0);
      });
#if 0
  addTargetMaterialization(
      [](mlir::OpBuilder &builder, Zmir::PendingType type,
         mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
        assert(inputs.size() == 1);

        if (auto compTyp =
                mlir::dyn_cast<Zmir::ComponentType>(inputs[0].getType())) {
          if (compTyp.getName().getValue() != "Component")
            return builder.create<Zmir::ReadSuperTransOp>(
                loc, Zmir::PendingType::get(builder.getContext()), inputs[0]);
        }
        return inputs[0];
      });
#endif
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
#if 0
  addArgumentMaterialization(
      [](mlir::OpBuilder &builder, Zmir::PendingType type,
         mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
        assert(inputs.size() == 1);

        if (auto compTyp =
                mlir::dyn_cast<Zmir::ComponentType>(inputs[0].getType())) {
          if (compTyp.getName().getValue() != "Component")
            return builder.create<Zmir::ReadSuperTransOp>(
                loc, Zmir::PendingType::get(builder.getContext()), inputs[0]);
        }
        return std::nullopt;
      });
#endif
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
