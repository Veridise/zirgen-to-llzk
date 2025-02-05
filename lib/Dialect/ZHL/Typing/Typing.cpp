#include "zklang/Dialect/ZHL/Typing/Typing.h"
#include "zklang/Dialect/ZHL/Typing/Rules.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/OpBindings.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

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

FailureOr<TypeBinding> applyRule(
    Operation *op, ZhlOpBindings &bindings, Scope &scope, const FrozenTypingRuleSet &rules,
    const TypingRule &rule, ArrayRef<TypeBinding> operands
) {
  // Check if the operation has regions and evaluate them first
  auto regions = op->getRegions();
  BlockScopesGuard regionScopes(scope);

  if (!std::all_of(regions.begin(), regions.end(), [&](auto &region) {
    auto valueBindings =
        rule.bindRegionArguments(ValueRange(region.getArguments()), op, operands, scope);
    if (failed(valueBindings)) {
      return false;
    }
    for (auto [value, type] : llvm::zip_equal(
             mlir::iterator_range(region.args_begin(), region.args_end()), *valueBindings
         )) {
      (void)bindings.addValue(value, type);
    }
    return succeeded(typeCheckRegion(region, regionScopes.get(), rules, bindings));
  })) {
    return failure();
  }

  return rule.typeCheck(op, operands, scope, regionScopes);
}

/// Handles the creation of the frame and linking it to the generated type binding if it succeeded.
FailureOr<TypeBinding> applyRuleWithFrame(
    Frame frame, Operation *op, ZhlOpBindings &bindings, Scope &scope,
    const FrozenTypingRuleSet &rules, const TypingRule &rule, ArrayRef<TypeBinding> operands
) {
  FrameScope frameScope(scope, frame);
  // Typecheck normally using a scope with the frame
  auto result = applyRule(op, bindings, frameScope, rules, rule, operands);

  if (failed(result)) {
    return failure();
  }
  // XXX: I may need to do more than a copy but we will see.
  auto finalBinding = result->ReplaceFrame(frame);
  finalBinding.markSlot(frame.getParentSlot());
  return finalBinding;
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
  for (auto &rule : rules) {
    if (failed(rule->match(op))) {
      continue;
    }
    auto frame = rule->allocate(scope.getCurrentFrame());
    FailureOr<TypeBinding> ruleResult =
        succeeded(frame) ? applyRuleWithFrame(*frame, op, bindings, scope, rules, *rule, *operands)
                         : applyRule(op, bindings, scope, rules, *rule, *operands);

    if (succeeded(ruleResult)) {
      return bindings.add(op, ruleResult);
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

using GraphInDegrees = llvm::StringMap<int>;

/// ZHL Component ops do not declare a symbol in MLIR's symbol tables machinery.
class ZhlSymbolTable {
public:
  explicit ZhlSymbolTable(Operation *op) : st(op) {}

  ComponentOp lookup(StringRef name) const {
    for (auto &region : st->getRegions()) {
      auto I = region.op_begin<ComponentOp>();
      auto E = region.op_end<ComponentOp>();
      for (; I != E; ++I) {
        auto op = *I;
        if (op.getName() == name) {
          return op;
        }
      }
    }
    return nullptr;
  }

private:
  Operation *st;
};

/// Fills the graph with all the components declared in ZHL. This ensures that
/// while iterating the map components with an indegree of 0 will still have an entry in the map.
void fillGraph(Operation *root, GraphInDegrees &indegrees) {
  root->walk([&](ComponentOp op) {
    llvm::dbgs() << "Adding " << op.getName() << " to the graph\n";
    indegrees[op.getName()];
  });
}

/// Sorts the components in topological order of declarations. This ensures that
/// any component that is used by another will be type checked before any dependants.
/// In ZIR this fine since recursion is forbidden, and thus the components must form a DAG.
/// If the given components do not form a DAG and have a cycle then this function returns failure
/// and reports an error back to the user.
LogicalResult
sortComponents(Operation *root, SmallVector<ComponentOp> &sortedComponents, ZhlSymbolTable &st) {
  assert(sortedComponents.empty() && "output vector must be empty");
  GraphInDegrees indegrees;
  fillGraph(root, indegrees);
  root->walk([&](ComponentOp op) {
    op->walk([&](GlobalOp global) {
      // We only add to the graph the component if its a name declared as a ZHL component op.
      // This excludes builtins that have bindings but not a ZHL representation because they are
      // generated directly in ZML.
      if (indegrees.contains(global.getName())) {
        llvm::dbgs() << "Component " << op.getName() << " depends on component " << global.getName()
                     << "\n";
        indegrees[global.getName()]++;
      }
    });
  });
  // To avoid unnecessary allocations
  sortedComponents.reserve(indegrees.size());

  // Prepare the initial queue for Kahn's algorithm with the components with indegree of 0
  SmallVector<ComponentOp> queue;
  for (auto &[name, indegreeCount] : indegrees) {
    if (indegreeCount == 0) {
      auto compOp = st.lookup(name);
      assert(compOp);
      llvm::dbgs() << "Adding " << name << " to the initial queue\n";
      queue.push_back(compOp);
    }
  }

  // Kahn's algorithm will find the topological order or fail if there is a cycle.
  while (!queue.empty()) {
    auto op = queue.front();
    queue.erase(queue.begin());
    llvm::dbgs() << "Adding " << op.getName() << " into the sorted vector\n";
    // Since its traverses the graph in reverse we need to add the elements to front of the vector
    sortedComponents.insert(sortedComponents.begin(), op);

    op->walk([&](GlobalOp global) {
      if (indegrees.contains(global.getName())) {
        indegrees[global.getName()]--;
        assert(indegrees[global.getName()] >= 0 && "More edges were removed than inserted");
        if (indegrees[global.getName()] == 0) {
          auto depOp = st.lookup(global.getName());
          assert(depOp);
          queue.push_back(depOp);
        }
      }
    });
  }

  // Check that all indegrees were remove (aka no cycles)
  for (auto &[name, indegree] : indegrees) {
    if (indegree > 0) {
      return root->emitError() << "Recursion cycle detected in component " << name;
    }
  }

  return success();
}

FailureOr<std::unique_ptr<ZhlOpBindings>>
typeCheck(Operation *root, TypeBindings &typeBindings, const FrozenTypingRuleSet &rules) {
  // Initialize what we need for type checking
  SymbolTable st(root);
  ZhlSymbolTable zhlSt(root);
  auto bindings = std::make_unique<ZhlOpBindings>();
  SmallVector<ComponentOp> sortedComponents;
  if (failed(sortComponents(root, sortedComponents, zhlSt))) {
    return failure();
  }
  // Run the typechecker
  for (auto op : sortedComponents) {
    llvm::dbgs() << "Type checking " << op.getName() << "\n";
    checkComponent(op, *bindings, typeBindings, rules);
  }

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
