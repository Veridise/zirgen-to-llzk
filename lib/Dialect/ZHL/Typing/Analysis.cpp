#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <unordered_set>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>
#include <zklang/Dialect/ZHL/Typing/OpBindings.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZHL/Typing/Typing.h> // IWYU pragma: keep
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>

using namespace mlir;

namespace zhl {

class zhl::ZIRTypeAnalysis::Impl {
public:
  virtual ~Impl() = default;
  virtual const mlir::FailureOr<TypeBinding> &getType(mlir::Operation *op) const = 0;
  virtual const mlir::FailureOr<TypeBinding> &getType(mlir::Value value) const = 0;
  virtual mlir::FailureOr<TypeBinding> getType(mlir::StringRef name) const = 0;
  virtual mlir::LogicalResult addType(mlir::Value value, const TypeBinding &binding) = 0;
  virtual const TypeBindings &getBindings() const = 0;
  virtual void print(llvm::raw_ostream &) const = 0;
  virtual operator mlir::LogicalResult() const = 0;
  virtual mlir::ArrayRef<TypeBinding *> getClosures() = 0;
};

namespace {

class DelegatedImpl : public zhl::ZIRTypeAnalysis::Impl {
public:
  DelegatedImpl(ZIRTypeAnalysis &analysis) : delegate(&analysis) {}

  const mlir::FailureOr<TypeBinding> &getType(mlir::Operation *op) const final {
    return delegate->getType(op);
  }

  const mlir::FailureOr<TypeBinding> &getType(mlir::Value value) const final {
    return delegate->getType(value);
  }

  mlir::FailureOr<TypeBinding> getType(mlir::StringRef name) const final {
    return delegate->getType(name);
  }

  mlir::LogicalResult addType(mlir::Value value, const TypeBinding &binding) final {
    return delegate->addType(value, binding);
  }

  const TypeBindings &getBindings() const override { return delegate->getBindings(); }

  void print(llvm::raw_ostream &os) const final { delegate->print(os); }

  operator mlir::LogicalResult() const final { return *delegate; }

  mlir::ArrayRef<TypeBinding *> getClosures() final { return delegate->getClosures(); }

private:
  ZIRTypeAnalysis *const delegate;
};

class ConcreteImpl : public zhl::ZIRTypeAnalysis::Impl {
public:
  explicit ConcreteImpl(mlir::ModuleOp module, mlir::OpBuilder &builder)
      : typeBindings(builder.getUnknownLoc()), failureRef(failure()), opBindings(nullptr) {
    std::unordered_set<std::string_view> definedNames;
    for (auto op : module.getOps<zirgen::Zhl::ComponentOp>()) {
      definedNames.insert(op.getName());
    }
    zml::addBuiltinBindings(typeBindings, definedNames);
    auto opBindingsResult = typeCheck(module, typeBindings, zhlTypingRules(typeBindings));
    if (failed(opBindingsResult)) {
      typeCheckingFailed = true;
    } else {
      opBindings = std::move(*opBindingsResult);
    }
  }

  const mlir::FailureOr<TypeBinding> &getType(mlir::Operation *op) const final {
    if (typeCheckingFailed) {
      return failureRef;
    }
    return opBindings->get(op);
  }

  const mlir::FailureOr<TypeBinding> &getType(mlir::Value value) const final {
    if (typeCheckingFailed) {
      return failureRef;
    }
    return opBindings->getValue(value);
  }

  mlir::FailureOr<TypeBinding> getType(mlir::StringRef name) const final {
    if (typeCheckingFailed) {
      return failureRef;
    }
    return typeBindings.MaybeGet(name);
  }

  mlir::LogicalResult addType(mlir::Value value, const TypeBinding &binding) final {
    if (typeCheckingFailed) {
      return mlir::failure();
    }

    return opBindings->addValue(value, binding);
  }

  const TypeBindings &getBindings() const override { return typeBindings; }

  void print(llvm::raw_ostream &os) const final {
    if (!typeCheckingFailed) {
      opBindings->print(os);
    } else {
      os << "type checking process failed and did not produce bindings";
    }
  }

  operator mlir::LogicalResult() const final {
    return typeCheckingFailed || failed(*opBindings) ? failure() : success();
  }

  mlir::ArrayRef<TypeBinding *> getClosures() final {
    if (!cachedClosures.has_value()) {
      cachedClosures = opBindings->getClosures();
    }
    return *cachedClosures;
  }

private:
  TypeBindings typeBindings;
  bool typeCheckingFailed = false;
  // Because the methods return references to a FailureOr
  mlir::FailureOr<TypeBinding> failureRef;
  std::unique_ptr<ZhlOpBindings> opBindings;
  std::optional<SmallVector<TypeBinding *>> cachedClosures;
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

mlir::LogicalResult ZIRTypeAnalysis::addType(mlir::Value value, const TypeBinding &binding) {
  return impl->addType(value, binding);
}

const TypeBindings &ZIRTypeAnalysis::getBindings() const { return impl->getBindings(); }
void ZIRTypeAnalysis::dump() const { print(llvm::dbgs()); }
void ZIRTypeAnalysis::print(llvm::raw_ostream &os) const { impl->print(os); }
ZIRTypeAnalysis::operator mlir::LogicalResult() const {
  if (impl == nullptr) {
    return mlir::failure();
  }
  return *impl;
}
mlir::ArrayRef<TypeBinding *> ZIRTypeAnalysis::getClosures() { return impl->getClosures(); }

} // namespace zhl
