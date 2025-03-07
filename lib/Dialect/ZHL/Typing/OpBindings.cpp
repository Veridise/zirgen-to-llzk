#include <zklang/Dialect/ZHL/Typing/OpBindings.h>

using namespace mlir;
using namespace llvm;

namespace zhl {

void OpBindings::dump() const { print(dbgs()); }

OpBindings::operator LogicalResult() const { return missingBindings() ? failure() : success(); }

void OpBindings::printBinding(raw_ostream &os, FailureOr<TypeBinding> type) const {
  os << " : ";
  if (succeeded(type)) {
    type->print(os);
  } else {
    os << "<type checking failed>";
  }
  os << "\n";
}
Value ValueBindings::opToKey(Operation *op) const { return op->getResult(0); }

void ValueBindings::validateOp(Operation *op) const {
  assert(op->getNumResults() == 1 && "expected an op with only 1 result");
}

void ValueBindings::printKey(Value k, raw_ostream &os) const { os << k; }

static Operation *findOpFromValue(Value v) {
  if (auto op = v.getDefiningOp()) {
    return op;
  }
  auto region = v.getParentRegion();
  if (!region) {
    return nullptr;
  }
  return region->getParentOp();
}

void ValueBindings::emitRemarks() const {
  for (auto &[val, binding] : *this) {
    auto op = findOpFromValue(val);
    auto valIsOp = val.getDefiningOp() != nullptr;
    assert(op);
    if (mlir::failed(binding)) {
      if (valIsOp) {
        op->emitError() << "type check failed";
      } else {
        op->emitError() << "type check for value " << val << " failed ";
      }
    } else {
      if (valIsOp) {
        op->emitRemark() << "type: " << *binding;
      } else {
        op->emitRemark() << "type for value " << val << ": " << *binding;
      }
    }
  }
}

Operation *StmtBindings::opToKey(Operation *op) const { return op; }

void StmtBindings::validateOp(Operation *op) const {
  assert(op->getNumResults() == 0 && "expected an op with no results");
}

void StmtBindings::printKey(Operation *k, raw_ostream &os) const { os << *k; }

void StmtBindings::emitRemarks() const {
  for (auto &[op, binding] : *this) {
    if (mlir::failed(binding)) {
      op->emitError() << "type check failed";
    } else {
      op->emitRemark() << "type: " << *binding;
    }
  }
}

FailureOr<TypeBinding> ZhlOpBindings::addValue(Value v, FailureOr<TypeBinding> type) {
  return values.addType(v, type);
}

FailureOr<TypeBinding> ZhlOpBindings::add(Operation *op, FailureOr<TypeBinding> type) {
  return (opIsValue(op)) ? values.add(op, type) : stmts.add(op, type);
}

const FailureOr<TypeBinding> &ZhlOpBindings::getValue(Value v) const { return values.getType(v); }

const FailureOr<TypeBinding> &ZhlOpBindings::get(Operation *op) const {
  return opIsValue(op) ? values.get(op) : stmts.get(op);
}

bool ZhlOpBindings::contains(Operation *op) const {
  return (opIsValue(op) && values.contains(op)) || stmts.contains(op);
}

bool ZhlOpBindings::containsValue(Value v) const { return values.containsKey(v); }

void ZhlOpBindings::print(raw_ostream &os) const {
  values.print(os);
  stmts.print(os);
}

bool ZhlOpBindings::missingBindings() const {
  return values.missingBindings() || stmts.missingBindings();
}

bool ZhlOpBindings::opIsValue(Operation *op) const { return op->getNumResults() == 1; }

mlir::SmallVector<TypeBinding *> ZhlOpBindings::getClosures() {
  SmallVector<TypeBinding *> closures;

  auto forEach = [&](auto &map) {
    for (auto &[_, binding] : map) {
      if (binding->hasClosure()) {
        closures.push_back(&*binding);
      }
    }
  };
  forEach(values.bindings);
  forEach(stmts.bindings);

  return closures;
}

void ZhlOpBindings::emitRemarks() const {
  values.emitRemarks();
  stmts.emitRemarks();
}

} // namespace zhl
