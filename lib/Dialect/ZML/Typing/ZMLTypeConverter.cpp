#include "zklang/Dialect/ZML/Typing/ZMLTypeConverter.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include <mlir/IR/BuiltinOps.h>
#include <optional>

using namespace zml;
using namespace zirgen;

ZMLTypeConverter::ZMLTypeConverter() {
  addConversion([](mlir::Type t) { return t; });

  addSourceMaterialization(
      [](mlir::OpBuilder &builder, Zhl::ExprType type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
    return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
  }
  );
  addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
    if (inputs.size() != 1) {
      return std::nullopt;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
  }
  );
  addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
    if (!mlir::isa<zml::ComponentType>(type)) {
      return std::nullopt;
    }
    if (auto compTyp = mlir::dyn_cast<zml::ComponentType>(inputs[0].getType())) {
      if (compTyp.getName().getValue() != "Component") {
        return builder.create<zml::SuperCoerceOp>(loc, type, inputs[0]);
      }
    }
    return std::nullopt;
  }
  );

  addArgumentMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
    return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
  }
  );
}
