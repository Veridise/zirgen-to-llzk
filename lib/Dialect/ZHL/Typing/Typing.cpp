#include "zklang/Dialect/ZHL/Typing/Typing.h"
#include "zklang/Dialect/ZHL/Typing/Rules.h"

using namespace mlir;
using namespace zirgen::Zhl;

namespace zhl {

FailureOr<TypeBinding>
typeCheckValue(Value v, ZhlOpBindings &bindings, Scope &scope, const FrozenTypingRuleSet &rules);
FailureOr<TypeBinding>
typeCheckOp(Operation *op, ZhlOpBindings &bindings, Scope &scope, const FrozenTypingRuleSet &rules);
FailureOr<std::vector<TypeBinding>> getOperandTypes(
    Operation *op, ZhlOpBindings &bindings, Scope &scope, const FrozenTypingRuleSet &rules
);

void checkComponent(
    ComponentOp comp, ZhlOpBindings &bindings, TypeBindings &typeBindings,
    const FrozenTypingRuleSet &rules
) {
  ComponentScope scope(comp, typeBindings);
  for (auto &op : comp.getRegion().getOps()) {
    (void)typeCheckOp(&op, bindings, scope, rules);
  }
}

FailureOr<TypeBinding>
typeCheckValue(Value v, ZhlOpBindings &bindings, Scope &scope, const FrozenTypingRuleSet &rules) {
  auto op = v.getDefiningOp();
  if (op) {
    return typeCheckOp(op, bindings, scope, rules);
  } else {
    if (bindings.containsValue(v)) {
      return bindings.getValue(v);
    }
    return failure();
  }
}

LogicalResult typeCheckRegion(
    Region &region, Scope &scope, const FrozenTypingRuleSet &rules, ZhlOpBindings &bindings
) {
  bool anyFailed = false;

  for (auto &op : region.getOps()) {
    auto result = typeCheckOp(&op, bindings, scope, rules);
    anyFailed = anyFailed || failed(result);
  }

  return anyFailed ? failure() : success();
}

class BlockScopesGuard {
public:
  explicit BlockScopesGuard(Scope &parent) : parent(parent) {}
  ~BlockScopesGuard() {
    for (auto scope : scopes) {
      delete scope;
    }
  }

  Scope &get() {
    auto scope = new BlockScope(parent);
    scopes.push_back(scope);
    return *scope;
  }

  operator mlir::ArrayRef<const Scope *>() { return scopes; }

private:
  Scope &parent;
  std::vector<Scope *> scopes;
};

template <typename Out, typename In1, typename In2>
inline void zip(In1 begin1, In1 end1, In2 begin2, Out out) {
  std::transform(begin1, end1, begin2, out, [](auto lhs, auto rhs) { return std::pair(lhs, rhs); });
}

FailureOr<TypeBinding> typeCheckOp(
    Operation *op, ZhlOpBindings &bindings, Scope &scope, const FrozenTypingRuleSet &rules
) {
  assert(op != nullptr);
  if (bindings.contains(op)) {
    return bindings.get(op);
  }
  auto operands = getOperandTypes(op, bindings, scope, rules);
  if (failed(operands)) {
    return failure();
  }
  // Check if the operation has regions and evaluate them first
  auto regions = op->getRegions();
  auto nRegions = op->getNumRegions();
  BlockScopesGuard regionScopes(scope);
  bool regionCheckFailed = false;
  for (unsigned i = 0; i < nRegions; i++) {
    for (auto &rule : rules) {
      if (failed(rule->match(op))) {
        continue;
      }

      auto valueBindings =
          rule->bindRegionArguments(ValueRange(regions[i].getArguments()), op, *operands, scope);
      if (failed(valueBindings)) {
        return failure();
      }
      assert(valueBindings->size() == regions[i].getNumArguments());
      std::vector<std::pair<Value, TypeBinding>> zipped;
      zip(regions[i].args_begin(), regions[i].args_end(), valueBindings->begin(),
          std::back_inserter(zipped));
      for (auto &[value, type] : zipped) {
        (void)bindings.addValue(value, type);
      }
      break;
    }

    regionCheckFailed = regionCheckFailed ||
                        failed(typeCheckRegion(regions[i], regionScopes.get(), rules, bindings));
  }
  if (regionCheckFailed) {
    return failure();
  }

  for (auto &rule : rules) {
    if (failed(rule->match(op))) {
      continue;
    }

    auto checkResult = rule->typeCheck(op, *operands, scope, regionScopes);
    if (succeeded(checkResult)) {
      return bindings.add(op, checkResult);
    }
  }
  return bindings.add(
      op, op->emitError() << "could not deduce the type of op '" << op->getName() << "'"
  );
}

FailureOr<std::vector<TypeBinding>> getOperandTypes(
    Operation *op, ZhlOpBindings &bindings, Scope &scope, const FrozenTypingRuleSet &rules
) {
  auto operands = op->getOperands();
  std::vector<TypeBinding> operandBindings;
  for (auto operand : operands) {
    auto r = typeCheckValue(operand, bindings, scope, rules);
    if (failed(r)) {
      return failure();
    }
    operandBindings.push_back(*r);
  }
  return operandBindings;
}

std::unique_ptr<ZhlOpBindings>
typeCheck(Operation *root, TypeBindings &typeBindings, const FrozenTypingRuleSet &rules) {
  SymbolTable st(root);
  std::unique_ptr<ZhlOpBindings> bindings = std::make_unique<ZhlOpBindings>();
  root->walk([&](ComponentOp op) { checkComponent(op, *bindings, typeBindings, rules); });
  return bindings;
}

FrozenTypingRuleSet zhlTypingRules(TypeBindings &bindings) {
  TypingRuleSet rules;
  rules.add<
      LiteralTypingRule, StringTypingRule, GlobalTypingRule, ParameterTypingRule, SuperTypingRule,
      GetGlobalTypingRule, ConstructTypingRule, ExternTypingRule, DeclareTypingRule,
      SubscriptTypeRule, SpecializeTypeRule, ConstrainTypeRule, DefineTypeRule, ArrayTypeRule,
      BlockTypeRule, ReduceTypeRule, RangeTypeRule, BackTypeRule, GenericParamTypeRule, MapTypeRule,
      LookupTypeRule, SwitchTypeRule, ConstructGlobalTypeRule>(bindings);

  return rules;
}

TypingRule::TypingRule(TypeBindings &bindings) : bindings(&bindings) {}
TypeBindings &TypingRule::getBindings() const { return *bindings; }
FrozenTypingRuleSet::FrozenTypingRuleSet(RuleSet &&rules) : rules(std::move(rules)) {}
FrozenTypingRuleSet::RuleSet::const_iterator FrozenTypingRuleSet::begin() const {
  return rules.begin();
}
FrozenTypingRuleSet::RuleSet::const_iterator FrozenTypingRuleSet::end() const {
  return rules.end();
}
TypingRuleSet::operator FrozenTypingRuleSet() { return FrozenTypingRuleSet(std::move(rules)); }
} // namespace zhl
