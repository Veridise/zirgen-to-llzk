#pragma once

#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>

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
};

/// Pure virtual class that defines an interface for generating types of a concrete dialect.
class BVTypeProvider {
public:
  virtual ~BVTypeProvider() = default;

  /// Returns the type that represents the memory for back-variables of the type passed as argument.
  /// Different implementations may have different preconditions for the passed type.
  virtual mlir::Type getMemType(mlir::Type) const = 0;

  /// Returns the type that represents the current cycle count.
  virtual mlir::Type getCycleType(mlir::MLIRContext *) const = 0;

  /// Returns the type held by a field. The operation argument is used to locate the field.
  virtual mlir::Type getFieldType(mlir::FlatSymbolRefAttr name, mlir::Operation *op) const = 0;
};

/// Holds the values required to support back-variables.
struct BVValues {
  mlir::Value Memory, CycleCount;
};

/// Inserts into the given function type the arguments for handling the back-variables and, if
/// given, inserts into the vector of locations the locations for the new parameters.
/// If the offset is given it will inject starting from that point.
/// The concrete types that are injected are provided by the provider. Returns the new function
/// type.
mlir::FunctionType injectBVFunctionParams(
    const BVTypeProvider &provider, mlir::FunctionType fn, unsigned offset = 0,
    std::vector<mlir::Location> *locs = nullptr
);

/// Returns the values of the back-variables data passed as arguments.
BVValues loadBVValues(mlir::ValueRange args, unsigned offset = 0);

/// Returns the back-variables data of the type used to represent memory that holds all the
/// back-variables for the given field.
BVValues
loadMemoryForField(BVValues parentValues, mlir::FlatSymbolRefAttr fieldName, mlir::Operation *field, const BVTypeProvider &provider, mlir::OpBuilder &);

/// Inserts the memory and cycle values into a list of arguments
void injectBVArgs(BVValues BV, mlir::SmallVectorImpl<mlir::Value> &args, unsigned offset = 0);

} // namespace lang::zir
