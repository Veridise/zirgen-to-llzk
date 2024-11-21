#include "Helpers.h"

using namespace zkc;

ComponentArity::ComponentArity() : isVariadic(false), paramCount(0) {}

ComponentArity zkc::getComponentConstructorArity(zirgen::Zhl::ComponentOp op) {
  ComponentArity arity;

  // Add locations for each index and keep them sorted
  std::map<uint32_t, mlir::Location> locsByIndex;
  for (auto paramOp : op.getOps<zirgen::Zhl::ConstructorParamOp>()) {
    arity.isVariadic = arity.isVariadic || paramOp.getVariadic();
    arity.paramCount = std::max({arity.paramCount, paramOp.getIndex() + 1});
    locsByIndex.insert({paramOp.getIndex(), paramOp.getLoc()});
  }

  // The iterator will be sorted since it's a `std::map`.
  std::transform(locsByIndex.begin(), locsByIndex.end(),
                 std::back_inserter(arity.locs),
                 [](auto &pair) { return pair.second; });

  return arity;
}
