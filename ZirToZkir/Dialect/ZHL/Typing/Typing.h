#pragma once

#include "OpBindings.h"
#include "Scope.h"
#include "TypeBindings.h"

namespace zhl {

class TypingRule {
public:
  explicit TypingRule(const TypeBindings &bindings);

  const TypeBindings &getBindings() const;

  virtual mlir::LogicalResult match(mlir::Operation *op) const = 0;

  virtual mlir::FailureOr<TypeBinding>
  typeCheck(mlir::Operation *, mlir::ArrayRef<TypeBinding>, Scope &, mlir::ArrayRef<const Scope *>)
      const = 0;

  virtual mlir::FailureOr<std::vector<TypeBinding>>
  bindRegionArguments(mlir::ValueRange, mlir::Operation *, mlir::ArrayRef<TypeBinding>, Scope &)
      const = 0;

private:
  const TypeBindings *bindings;
};

template <typename Op> class OpTypingRule : public TypingRule {
public:
  using TypingRule::TypingRule;

  mlir::LogicalResult match(mlir::Operation *op) const override {
    if (auto o = mlir::dyn_cast<Op>(op)) {
      return match(o);
    }
    return mlir::failure();
  }

  mlir::FailureOr<TypeBinding> typeCheck(
      mlir::Operation *op, mlir::ArrayRef<TypeBinding> operands, Scope &scope,
      mlir::ArrayRef<const Scope *> regionScopes
  ) const override {
    assert(op->getParentOfType<zirgen::Zhl::ComponentOp>() == scope.getOp());
    if (auto o = mlir::dyn_cast<Op>(op)) {
      return typeCheck(o, operands, scope, regionScopes);
    }
    return mlir::failure();
  }

  mlir::FailureOr<std::vector<TypeBinding>> bindRegionArguments(
      mlir::ValueRange args, mlir::Operation *op, mlir::ArrayRef<TypeBinding> operands, Scope &scope
  ) const override {
    if (auto o = mlir::dyn_cast<Op>(op)) {
      return bindRegionArguments(args, o, operands, scope);
    }
    return mlir::failure();
  }

  virtual mlir::LogicalResult match(Op op) const { return mlir::success(); }

  virtual mlir::FailureOr<TypeBinding>
  typeCheck(Op op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const = 0;

  virtual mlir::FailureOr<std::vector<TypeBinding>>
  bindRegionArguments(mlir::ValueRange args, Op, mlir::ArrayRef<TypeBinding>, Scope &) const {
    assert(args.size() == 0); // An op that has arguments must override this method
    return std::vector<TypeBinding>();
  }
};

class FrozenTypingRuleSet {
public:
  using RuleSet = std::vector<std::unique_ptr<TypingRule>>;

  explicit FrozenTypingRuleSet(RuleSet &&rules);

  RuleSet::const_iterator begin() const;
  RuleSet::const_iterator end() const;

private:
  RuleSet rules;
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

  operator FrozenTypingRuleSet();

private:
  std::vector<std::unique_ptr<TypingRule>> rules;
};

std::unique_ptr<ZhlOpBindings>
typeCheck(mlir::Operation *, TypeBindings &, const FrozenTypingRuleSet &);
FrozenTypingRuleSet zhlTypingRules(const TypeBindings &);

} // namespace zhl
