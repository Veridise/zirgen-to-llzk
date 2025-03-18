#pragma once

#include <cstdint>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>

namespace lang::zir {

struct BVConstants {
#ifndef NDEBUG
  /// Cap the number of cycles to a positive signed 8 bit integer (in debug mode) since the
  /// operation for computing the visited cycle is a subtraction.
  static constexpr uint64_t CYCLE_COUNTER_BITWIDTH = 7;
#else
  /// Cap the number of cycles to a positive signed 64 bit integer since the operation for computing
  /// the visited cycle is a subtraction.
  static constexpr uint64_t CYCLE_COUNTER_BITWIDTH = 63;
#endif
  static constexpr uint64_t CYCLES = 1ul << CYCLE_COUNTER_BITWIDTH;

  /// Base position for the parameter that holds the back-variables memory.
  static constexpr uint8_t MEMORY_ARG_NO = 0;
  /// Base position for the parameter that holds the value of the current cycle.
  static constexpr uint8_t CYCLE_ARG_NO = 1;

  /// Number of added arguments to support back-variables in a function.
  static constexpr uint8_t ADDED_ARGS = 2;
};

/// Pure virtual class that defines an interface for generating types of a concrete dialect.
class BVDialectHelper {
public:
  virtual ~BVDialectHelper() = default;

  /// Returns the type that represents the memory for back-variables of the type passed as argument.
  /// Different implementations may have different preconditions for the passed type.
  virtual mlir::Type getMemType(mlir::Type) const = 0;

  /// Returns the type that represents the current cycle count.
  virtual mlir::Type getCycleType(mlir::MLIRContext *) const = 0;

  /// Returns a Value that represents a constant value of the cycle type.
  virtual mlir::Value getCycleConstant(mlir::OpBuilder &, uint64_t, mlir::Location) const = 0;

  /// Returns a Value that represents memory allocated to hold back-variables
  virtual mlir::Value allocateMem(mlir::OpBuilder &, mlir::Type, mlir::Location) const = 0;

  /// Creates ops to read the n-th value of an array and returns it.
  virtual mlir::Value readArray(
      mlir::Type, mlir::Value array, mlir::Value n, mlir::OpBuilder &, mlir::Location
  ) const = 0;

  /// Creates ops to write a value into the n-th index of an array.
  virtual void writeArray(
      mlir::Value array, mlir::Value n, mlir::Value in, mlir::OpBuilder &, mlir::Location
  ) const = 0;

  /// Creates ops to read the content of a field.
  virtual mlir::Value readField(
      mlir::Type, mlir::FlatSymbolRefAttr name, mlir::Value src, mlir::OpBuilder &, mlir::Location
  ) const = 0;

  /// Given a function returns the type that represents the component
  virtual mlir::Type deduceComponentType(mlir::FunctionOpInterface) const = 0;

  /// Creates ops to subtract two values.
  virtual mlir::Value
  subtractValues(mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder &, mlir::Location) const = 0;
};

/// Opaque struct that holds the values required to support back-variables.
/// It is opaque to enforce the separation between the back-variables data and the rest of the
/// lowering. This prevents spillage which will help when we revisit this module in the future.
class BVValues {
public:
  struct Impl;
  std::unique_ptr<Impl> impl;

  BVValues(Impl *);
  ~BVValues();
};

/// Inserts into the given function type the arguments for handling the back-variables and, if
/// given, inserts into the vector of locations the locations for the new parameters.
/// If the offset is given it will inject starting from that point.
/// The concrete types that are injected are provided by the helper. Returns the new function
/// type.
mlir::FunctionType injectBVFunctionParams(
    const BVDialectHelper &, mlir::FunctionType, unsigned offset = 0,
    llvm::SmallVectorImpl<mlir::Location> *locs = nullptr
);

/// Returns a new ArrayRef that hides the injected back-variable types.
mlir::ArrayRef<mlir::Type> hideInjectedBVTypes(mlir::ArrayRef<mlir::Type>);

/// Returns the values of the back-variables given to a function.
BVValues loadBVValues(const BVDialectHelper &, mlir::FunctionOpInterface, unsigned offset = 0);

/// Returns the back-variables data of the type used to represent memory that holds all the
/// back-variables for the given field.
BVValues loadMemoryForField(
    BVValues &parentValues, mlir::FlatSymbolRefAttr fieldName, mlir::Type fieldType,
    const BVDialectHelper &, mlir::OpBuilder &, mlir::Location
);

/// Inserts the memory and cycle values into a list of arguments
void injectBVArgs(BVValues &, mlir::SmallVectorImpl<mlir::Value> &args, unsigned offset = 0);

/// Creates instructions to read a back-variable.
mlir::Value readBackVariable(
    BVValues &, mlir::FlatSymbolRefAttr fieldName, mlir::Type fieldType, mlir::Value distance,
    const BVDialectHelper &, mlir::OpBuilder &, mlir::Location
);

} // namespace lang::zir
