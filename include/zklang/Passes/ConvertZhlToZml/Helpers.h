#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include <cstdint>
#include <vector>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>

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

/// Creates a slot for storing a result inside a component's body.
mlir::FlatSymbolRefAttr createSlot(
    zhl::ComponentSlot *slot, mlir::OpBuilder &builder, Zmir::ComponentInterface component,
    mlir::Location loc
);

/// Stores a value into the field defined by the slotName symbol and immediately reads it back.
/// Returns the value read from the field.
mlir::Value storeAndLoadSlot(
    mlir::Value value, mlir::FlatSymbolRefAttr slotName, mlir::Type slotType, mlir::Location loc,
    mlir::Type compType, mlir::OpBuilder &builder, mlir::Value
);

/// Helper for creating the ops that represent the call to a component's constructor.
/// If the associated binding is linked to a frame slot it also creates the field and writes the
/// result into it.
class CtorCallBuilder {
public:
  static mlir::FailureOr<CtorCallBuilder> Make(
      mlir::Operation *op, mlir::Value value, const zhl::ZIRTypeAnalysis &typeAnalysis,
      mlir::OpBuilder &builder, mlir::Value self
  );

  static mlir::FailureOr<CtorCallBuilder> Make(
      mlir::Operation *op, const zhl::TypeBinding &binding, mlir::OpBuilder &builder,
      mlir::Value self
  );

  /// Generates the ops that represent the construction of a component. Fetches a reference to
  /// the constructor and calls it. If the binding is linked to a frame slot creates a field in the
  /// component and writes the result into it. If a field is created returns the Value of reading
  /// the field, otherwise returns the value returned by the constructor call. The returned value is
  /// also used as argument to create a constrain call op that represents the call to `@constrain`
  /// in LLZK.
  mlir::Value build(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args);
  mlir::FunctionType getCtorType() const;
  const zhl::TypeBinding &getBinding() const;
  Zmir::ComponentInterface getCalleeComp() const;
  Zmir::ComponentInterface getCallerComp() const;

private:
  CtorCallBuilder(
      mlir::FunctionType type, const zhl::TypeBinding &binding, Zmir::ComponentInterface callee,
      Zmir::ComponentInterface caller, mlir::Value self, bool builtin
  );

  mlir::FunctionType ctorType;
  const zhl::TypeBinding compBinding;
  bool isBuiltin;
  Zmir::ComponentInterface calleeComponentOp, callerComponentOp;
  mlir::Value self;
};

} // namespace zkc
