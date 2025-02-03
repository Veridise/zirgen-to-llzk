#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include <cstdint>
#include <vector>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>

namespace zkc {

struct ComponentArity {
  /// The last argument of a ZIR component can be variadic.
  bool isVariadic;
  uint32_t paramCount;
  std::vector<mlir::Location> locs;

  ComponentArity();
};

/// Searches all Zhl::ConstructorParamOp in the component
/// and returns the largest index declared by the ops.
ComponentArity getComponentConstructorArity(zirgen::Zhl::ComponentOp);

mlir::FlatSymbolRefAttr createTempField(
    mlir::Location loc, mlir::Type type, mlir::OpBuilder &builder, Zmir::ComponentInterface op
);

/// Creates a temporary field to store the value and a sequence of reads and writes
/// that disconnect the value creation from its users.
mlir::Operation *storeValueInTemporary(
    mlir::Location loc, Zmir::ComponentOp callerComp, mlir::Type fieldType, mlir::Value value,
    mlir::ConversionPatternRewriter &rewriter
);

/// Finds the definition of the callee component. If the
/// component was defined before the current operation wrt the physical order of
/// the file then its defined by a ZML ComponentInterface op, if it hasn't been
/// converted yet it is still a ZHL Component op. If the name could not be found
/// in either form returns nullptr.
mlir::Operation *findCallee(mlir::StringRef name, mlir::ModuleOp root);

/// Returns true if the operation is a component and has the builtin attribute
bool calleeIsBuiltin(mlir::Operation *op);

/// Helper for creating the ops that represent the call to a component's constructor
class CtorCallBuilder {
public:
  static mlir::FailureOr<CtorCallBuilder> Make(
      mlir::Operation *op, mlir::Value value, const zhl::ZIRTypeAnalysis &typeAnalysis,
      mlir::OpBuilder &builder
  );

  mlir::Value build(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args);
  mlir::FunctionType getCtorType() const;

private:
  CtorCallBuilder(mlir::FunctionType type, const zhl::TypeBinding &binding, bool builtin);

  mlir::FunctionType ctorType;
  const zhl::TypeBinding compBinding;
  bool isBuiltin;
};

} // namespace zkc
