#include "Analysis.h"
#include "OpBindings.h"
#include "TypeBindings.h"
#include "Typing.h" // IWYU pragma: keep
#include "ZirToZkir/Dialect/ZMIR/BuiltIns/BuiltIns.h"
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <unordered_set>

namespace zhl {

class zhl::ZIRTypeAnalysis::Impl {
public:
  virtual const mlir::FailureOr<TypeBinding> &getType(mlir::Operation *op) const = 0;
  virtual const mlir::FailureOr<TypeBinding> &getType(mlir::Value value) const = 0;
  virtual mlir::FailureOr<TypeBinding> getType(mlir::StringRef name) const = 0;
  virtual void print(llvm::raw_ostream &) const = 0;
  virtual operator mlir::LogicalResult() const = 0;
};

namespace {

class DelegatedImpl : public zhl::ZIRTypeAnalysis::Impl {
public:
  DelegatedImpl(ZIRTypeAnalysis &delegate) : delegate(&delegate) {}

  const mlir::FailureOr<TypeBinding> &getType(mlir::Operation *op) const final {
    return delegate->getType(op);
  }

  const mlir::FailureOr<TypeBinding> &getType(mlir::Value value) const final {
    return delegate->getType(value);
  }

  mlir::FailureOr<TypeBinding> getType(mlir::StringRef name) const final {
    return delegate->getType(name);
  }

  void print(llvm::raw_ostream &os) const final { delegate->print(os); }

  operator mlir::LogicalResult() const final { return *delegate; }

private:
  const ZIRTypeAnalysis *delegate;
};

class ConcreteImpl : public zhl::ZIRTypeAnalysis::Impl {
public:
  explicit ConcreteImpl(mlir::ModuleOp module, mlir::OpBuilder &builder)
      : typeBindings(builder), opBindings(nullptr) {
    std::unordered_set<std::string_view> definedNames;
    for (auto op : module.getOps<zirgen::Zhl::ComponentOp>()) {
      definedNames.insert(op.getName());
    }
    zkc::Zmir::addBuiltinBindings(typeBindings, definedNames);
    opBindings = typeCheck(module, typeBindings, zhlTypingRules(typeBindings));
  }

  const mlir::FailureOr<TypeBinding> &getType(mlir::Operation *op) const final {
    return opBindings->get(op);
  }

  const mlir::FailureOr<TypeBinding> &getType(mlir::Value value) const final {
    return opBindings->getValue(value);
  }

  mlir::FailureOr<TypeBinding> getType(mlir::StringRef name) const final {
    return typeBindings.MaybeGet(name);
  }

  void print(llvm::raw_ostream &os) const final { opBindings->print(os); }

  operator mlir::LogicalResult() const final { return *opBindings; }

private:
  TypeBindings typeBindings;
  std::unique_ptr<ZhlOpBindings> opBindings;
};
} // namespace

template <typename ImplType, typename... Args>
std::unique_ptr<zhl::ZIRTypeAnalysis::Impl> makeImpl(Args &&...args) {
  return std::make_unique<ImplType>(std::forward<Args>(args)...);
}

ZIRTypeAnalysis::ZIRTypeAnalysis(mlir::Operation *op, mlir::AnalysisManager &am) : impl(nullptr) {
  if (auto mod = mlir::dyn_cast<mlir::ModuleOp>(op)) {
    mlir::OpBuilder builder(mod.getBodyRegion());
    impl = makeImpl<ConcreteImpl>(mod, builder);
  } else {
    auto parent = am.getCachedParentAnalysis<ZIRTypeAnalysis>(op->getParentOp());
    assert(parent.has_value());
    impl = makeImpl<DelegatedImpl>(*parent);
  }
}

ZIRTypeAnalysis::~ZIRTypeAnalysis() = default;

const mlir::FailureOr<TypeBinding> &ZIRTypeAnalysis::getType(mlir::Operation *op) const {
  return impl->getType(op);
}
const mlir::FailureOr<TypeBinding> &ZIRTypeAnalysis::getType(mlir::Value value) const {
  return impl->getType(value);
}

mlir::FailureOr<TypeBinding> ZIRTypeAnalysis::getType(mlir::StringRef name) const {
  return impl->getType(name);
}
void ZIRTypeAnalysis::dump() const { print(llvm::dbgs()); }
void ZIRTypeAnalysis::print(llvm::raw_ostream &os) const { impl->print(os); }
ZIRTypeAnalysis::operator mlir::LogicalResult() const {
  if (impl == nullptr) {
    return mlir::failure();
  }
  return *impl;
}

} // namespace zhl
