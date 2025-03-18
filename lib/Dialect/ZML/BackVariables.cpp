#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Dialect/ZML/Utils/BackVariables.h>

using namespace zml;
using namespace mlir;

Value ZML_BVDialectHelper::getCycleConstant(OpBuilder &builder, uint64_t value, mlir::Location loc)
    const {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

Value ZML_BVDialectHelper::allocateMem(OpBuilder &builder, Type arrType, mlir::Location loc) const {
  assert(mlir::isa<ComponentType>(arrType));
  assert(mlir::cast<ComponentType>(arrType).isConcreteArray());
  return builder.create<AllocArrayOp>(loc, arrType);
}

Value ZML_BVDialectHelper::readArray(
    Type type, Value array, Value n, OpBuilder &builder, Location loc
) const {
  return builder.create<ReadArrayOp>(loc, type, array, n);
}

void ZML_BVDialectHelper::writeArray(
    Value array, Value n, Value in, OpBuilder &builder, Location loc
) const {
  builder.create<WriteArrayOp>(loc, array, n, in);
}

Value ZML_BVDialectHelper::readField(
    Type type, FlatSymbolRefAttr name, Value src, OpBuilder &builder, Location loc
) const {
  return builder.create<ReadFieldOp>(loc, type, src, name);
}

Type ZML_BVDialectHelper::deduceComponentType(FunctionOpInterface func) const {
  assert(func.getNumResults() == 1);
  return func.getResultTypes().front();
}
