// Copyright 2024 Veridise, Inc.

#include "ZirToZkir/Dialect/ZHL/Typing/PassDetail.h"
#include "ZirToZkir/Dialect/ZHL/Typing/TypeBindings.h"
#include "ZirToZkir/Dialect/ZMIR/BuiltIns/BuiltIns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include <cassert>
#include <llvm/Support/Debug.h>

using namespace zirgen::Zhl;
using namespace mlir;

namespace zhl {

class TypingRule {
public:
  explicit TypingRule(const TypeBindings &bindings) : bindings(&bindings) {}

  const TypeBindings &getBindings() const { return *bindings; }
  virtual mlir::FailureOr<TypeBinding>
  typeCheck(mlir::Operation *, mlir::ArrayRef<TypeBinding>) const = 0;

private:
  const TypeBindings *bindings;
};

class TypingResult {
public:
  void add(mlir::Operation *op, mlir::FailureOr<TypeBinding> type) { results.insert({op, type}); }

  const mlir::FailureOr<TypeBinding> &get(mlir::Operation *op) const {
    if (results.find(op) == results.end()) {
      return failed;
    }
    return results.at(op);
  }

  void dump() const { print(llvm::dbgs()); }

  void print(llvm::raw_ostream &os) const {
    for (auto &[op, type] : results) {
      op->print(os);
      os << " : ";
      if (mlir::succeeded(type)) {
        type->print(os);
      } else {
        os << "<type checking failed>";
      }
      os << "\n";
    }
  }

  operator LogicalResult() const {
    return std::any_of(
               results.begin(), results.end(), [](auto &pair) { return mlir::failed(pair.second); }
           )
               ? mlir::failure()
               : mlir::success();
  }

private:
  mlir::FailureOr<TypeBinding> failed;
  std::map<mlir::Operation *, mlir::FailureOr<TypeBinding>> results;
};

class TypingRuleSet {
public:
  template <
      typename... Rules, typename ConstructorArg, typename... ConstructorArgs,
      typename = std::enable_if_t<sizeof...(Rules) != 0>>
  inline void add(ConstructorArg &&arg, ConstructorArgs &&...args) {
    (void(rules.push_back(std::make_unique<Rules>(
         std::forward<ConstructorArg>(arg), std::forward<ConstructorArgs>(args)...
     ))),
     ...);
  }

  template <typename... Rules> inline void add() {
    (void(rules.push_back(std::make_unique<Rules>())), ...);
  }

  template <typename... Ops> TypingResult check(mlir::Operation *root) {
    TypingResult result;
    root->walk([&](mlir::Operation *op) {
      if (!mlir::isa<Ops...>(op)) {
        return;
      }

      for (auto &rule : rules) {
        std::vector<TypeBinding> operands; // TODO Extract operands
        auto checkResult = rule->typeCheck(op, operands);
        if (mlir::succeeded(checkResult)) {
          result.add(op, checkResult);
          return;
        }
      }
      result.add(
          op,
          op->emitOpError() << "could not deduce the type of requested op '" << op->getName() << "'"
      );
    });
    return result;
  }

private:
  std::vector<std::unique_ptr<TypingRule>> rules;
};

template <typename Op> class OpTypingRule : public TypingRule {
public:
  using TypingRule::TypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(mlir::Operation *op, mlir::ArrayRef<TypeBinding> operands) const override {
    if (auto o = mlir::dyn_cast<Op>(op)) {
      return typeCheck(o, operands);
    }
    return mlir::failure();
  }

  virtual mlir::FailureOr<TypeBinding>
  typeCheck(Op op, mlir::ArrayRef<TypeBinding> operands) const = 0;
};

class LiteralTypingRule : public OpTypingRule<LiteralOp> {
public:
  using OpTypingRule<LiteralOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding> typeCheck(LiteralOp op, mlir::ArrayRef<TypeBinding>) const override {
    return getBindings().Get("Val");
  }
};

class StringTypingRule : public OpTypingRule<StringOp> {
public:
  using OpTypingRule<StringOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding> typeCheck(StringOp op, mlir::ArrayRef<TypeBinding>) const override {
    return getBindings().Get("String");
  }
};

class GlobalTypingRule : public OpTypingRule<GlobalOp> {
public:
  using OpTypingRule<GlobalOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding> typeCheck(GlobalOp op, mlir::ArrayRef<TypeBinding>) const override {
    auto binding = getBindings().MaybeGet(op.getName());
    if (mlir::failed(binding)) {
      return op->emitOpError() << "type '" << op.getName() << "' was not found";
    }
    return binding;
  }
};

namespace {

class PrintTypeBindingsPass : public PrintTypeBindingsBase<PrintTypeBindingsPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    TypeBindings bindings;
    zkc::Zmir::addBuiltinBindings(bindings);

    TypingRuleSet rules;
    rules.add<LiteralTypingRule, StringTypingRule, GlobalTypingRule>(bindings);

    auto result = rules.check<LiteralOp, StringOp, GlobalOp>(mod);
    result.dump();
    if (mlir::failed(result)) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPrintTypeBindingsPass() {
  return std::make_unique<PrintTypeBindingsPass>();
}

} // namespace zhl
