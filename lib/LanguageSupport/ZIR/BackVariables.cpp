#include <cassert>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <zklang/LanguageSupport/ZIR/BackVariables.h>

using namespace lang::zir;
using namespace mlir;

struct lang::zir::BVValues::Impl {
  Value Memory, CycleCount;
  Type Component;
};

BVValues::BVValues(Impl *I) : impl(std::unique_ptr<Impl>(I)) {}

// Definition lives here because otherwise std::unique_ptr will not be able to generate the
// destructor outside this module since Impl is defined here.
BVValues::~BVValues() = default;

template <typename Col> Col::iterator computeIterator(Col &col, unsigned offset) {
  if (col.empty()) {
    return col.end();
  }
  return std::next(col.begin(), offset);
}

mlir::FunctionType lang::zir::injectBVFunctionParams(
    const BVDialectHelper &provider, mlir::FunctionType fn, unsigned offset,
    llvm::SmallVectorImpl<mlir::Location> *locs
) {
  assert(fn.getNumResults() == 1);
  assert(offset <= fn.getNumInputs());
  if (locs) {
    assert(fn.getNumInputs() == locs->size());
  }

  Builder builder(fn.getContext());
  SmallVector<Type> inputs(fn.getInputs());
  inputs.insert(
      computeIterator(inputs, offset),
      {provider.getMemType(fn.getResult(0)), provider.getCycleType(builder.getContext())}
  );

  if (locs) {
    locs->insert(
        computeIterator(*locs, offset), {builder.getUnknownLoc(), builder.getUnknownLoc()}
    );
  }

  return builder.getFunctionType(inputs, fn.getResults());
}

BVValues lang::zir::loadBVValues(
    const BVDialectHelper &helper, mlir::FunctionOpInterface func, unsigned offset
) {
  return BVValues(new BVValues::Impl{
      .Memory = func.getArguments()[offset],
      .CycleCount = func.getArguments()[offset + 1],
      .Component = helper.deduceComponentType(func)
  });
}

BVValues lang::zir::loadMemoryForField(
    BVValues &parentValues, mlir::FlatSymbolRefAttr fieldName, Type fieldType,
    const BVDialectHelper &provider, mlir::OpBuilder &builder, Location loc
) {
  // auto fieldType = provider.getFieldType(fieldName, field);
  // assert(fieldType);
  auto memType = provider.getMemType(fieldType);
  assert(memType);

  auto lb = provider.getCycleConstant(builder, 0, loc);
  auto ub = parentValues.impl->CycleCount;
  auto stride = provider.getCycleConstant(builder, 1, loc);

  auto mem = provider.allocateMem(builder, memType, loc);

  auto copyLoop = builder.create<scf::ForOp>(
      loc, lb, ub, stride, ValueRange{mem},
      [parentMem = parentValues.impl->Memory, parentType = parentValues.impl->Component, fieldName,
       fieldType, &provider](OpBuilder &loopBuilder, Location loopLoc, Value iv, ValueRange args) {
    auto arrValue = provider.readArray(parentType, parentMem, iv, loopBuilder, loopLoc);
    auto backValue = provider.readField(fieldType, fieldName, arrValue, loopBuilder, loopLoc);
    provider.writeArray(args[0], iv, backValue, loopBuilder, loopLoc);
    loopBuilder.create<scf::YieldOp>(loopLoc, args[0]);
  }
  );

  return BVValues(new BVValues::Impl{
      .Memory = copyLoop.getResult(0),
      .CycleCount = parentValues.impl->CycleCount,
      .Component = fieldType
  });
}

void lang::zir::injectBVArgs(
    BVValues &BV, mlir::SmallVectorImpl<mlir::Value> &args, unsigned offset
) {
  args.insert(std::next(args.begin(), offset + 1), {BV.impl->Memory, BV.impl->CycleCount});
}
