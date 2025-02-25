#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

namespace zml {

namespace extval {

/// This interface exposes the logic for representing extending finite field elements for a
/// particular field.
/// Implementations of this interface define how the Ext* family of lowerable ops are converted
/// into lower level operations. Implementations are meant to use LLZK since this lowering is
/// intended to happen in the conversion pass from ZML to LLZK.
class BaseConverter {
public:
  class TypeHelper;

  /// A simple wrapper that implements operators +, -, * for readability.
  /// It is meant to be used to define a static set of operations in a readable manner while
  /// instantiating the required operations on the fly via the TypeHelper and the OpBuilder.
  class ValueWrap {
    mlir::Value val;
    mlir::OpBuilder *builder;
    const TypeHelper *helper;

  public:
    ValueWrap(mlir::Value Val, mlir::OpBuilder &Builder, const TypeHelper &Helper);
    ValueWrap(uint64_t Val, mlir::OpBuilder &Builder, const TypeHelper &Helper);

    operator mlir::Value() const;
    ValueWrap inv();
    ValueWrap operator+(const ValueWrap &other);
    ValueWrap operator-(const ValueWrap &other);
    ValueWrap operator-();
    ValueWrap operator*(const ValueWrap &other);
  };

  /// A bridge interface that handles the conversion of types and the creation of the operations
  /// for working with them
  class TypeHelper {
  public:
    virtual ~TypeHelper() = default;

    /// Returns the Type that represents the extended element at the lower level.
    virtual mlir::Type createArrayRepr(mlir::MLIRContext *) const = 0;
    /// Collects a set of values into a single value, usually via an array creation op.
    virtual mlir::Value
    collectValues(mlir::ValueRange, mlir::Location, mlir::OpBuilder &) const = 0;
    /// Reads each element of the array that represents the extended field element and wraps them in
    /// ValueWrap instances.
    virtual mlir::SmallVector<ValueWrap>
    wrapArrayValues(mlir::Value v, mlir::OpBuilder &builder) const = 0;
    /// Ensures that the provided type matches the low level type representation of an ExtVal.
    void assertIsValidRepr(mlir::Type t) const;
    void assertIsValidRepr(mlir::Value v) const;

    /// Creates an operation that represents addition between elements of the array representation
    virtual mlir::Value createAddOp(mlir::Value, mlir::Value, mlir::OpBuilder &) const = 0;
    /// Creates an operation that represents subtraction between elements of the array
    /// representation
    virtual mlir::Value createSubOp(mlir::Value, mlir::Value, mlir::OpBuilder &) const = 0;
    /// Creates an operation that represents multiplication between elements of the array
    /// representation
    virtual mlir::Value createMulOp(mlir::Value, mlir::Value, mlir::OpBuilder &) const = 0;
    /// Creates an operation that represents negation of elements of the array representation
    virtual mlir::Value createNegOp(mlir::Value, mlir::OpBuilder &) const = 0;
    /// Creates an operation that represents the inverse of elements of the array representation
    virtual mlir::Value createInvOp(mlir::Value, mlir::OpBuilder &) const = 0;
    /// Creates an operation that represents a literal value of the inner type of the array
    /// representation
    virtual mlir::Value createLitOp(uint64_t, mlir::OpBuilder &) const = 0;

    /// Creates an operation that returns a boolean value indicating if the input value is equal to
    /// 0 or not
    virtual mlir::Value createIszOp(mlir::Value, mlir::OpBuilder &) const = 0;
    /// Creates an assert operation
    virtual mlir::Operation *
    createAssertOp(mlir::Value, mlir::StringAttr, mlir::OpBuilder &) const = 0;
    /// Creates a logical and operation
    virtual mlir::Value createAndOp(mlir::Value, mlir::Value, mlir::OpBuilder &) const = 0;
  };

  BaseConverter(const TypeHelper &);

  virtual ~BaseConverter() = default;
  virtual mlir::Value lowerOp(
      zml::ExtAddOp op, zml::ExtAddOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const = 0;
  virtual mlir::Value lowerOp(
      zml::ExtSubOp op, zml::ExtSubOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const = 0;
  virtual mlir::Value lowerOp(
      zml::ExtMulOp op, zml::ExtMulOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const = 0;
  virtual mlir::Value lowerOp(
      zml::ExtInvOp op, zml::ExtInvOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const = 0;
  virtual mlir::Value lowerOp(
      zml::MakeExtOp op, zml::MakeExtOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const = 0;

  const TypeHelper &getTypeHelper() const { return *helper; }

private:
  const TypeHelper *helper;
};

} // namespace extval

} // namespace zml
