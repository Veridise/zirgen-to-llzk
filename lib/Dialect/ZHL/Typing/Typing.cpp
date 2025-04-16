//===- Typing.cpp - Type checking -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/OpBindings.h>
#include <zklang/Dialect/ZHL/Typing/Rules.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZHL/Typing/Typing.h>

#define DEBUG_TYPE "type-checker"

#ifndef NDEBUG
static uint8_t ident;
static mlir::StringRef loggingScope = "";

inline llvm::raw_ostream &logLine() {
  if (loggingScope.empty()) {
    llvm::dbgs() << "[Typing] ";
  } else {
    llvm::dbgs() << "[Typing(" << loggingScope << ")] ";
  }
  llvm::dbgs().indent(ident * 4);
  return llvm::dbgs();
}
#endif

using namespace mlir;
using namespace zirgen::Zhl;

namespace zhl {

namespace {

class BlockScopesGuard {
public:
  explicit BlockScopesGuard(Scope &Parent, TypeBindings &Bindings)
      : parent(&Parent), bindings(&Bindings) {}
  ~BlockScopesGuard() {
    for (auto scope : scopes) {
      delete scope;
    }
  }

  Scope &get() {
    auto scope = new BlockScope(*parent, *bindings);
    scopes.push_back(scope);
    return *scope;
  }

  operator mlir::ArrayRef<const Scope *>() { return scopes; }

private:
  Scope *parent;
  TypeBindings *bindings;
  std::vector<Scope *> scopes;
};

class TypeCheckingEnv {
public:
  TypeCheckingEnv(ZhlOpBindings &Bindings, const FrozenTypingRuleSet &Rules, TypeBindings &TB)
      : bindings(&Bindings), rules(&Rules), typeBindings(&TB) {}

  ZhlOpBindings &getBindings() { return *bindings; }

  const FrozenTypingRuleSet &getRules() { return *rules; }

  TypeBindings &getTypeBindings() { return *typeBindings; }

private:
  ZhlOpBindings *bindings;
  const FrozenTypingRuleSet *rules;
  TypeBindings *typeBindings;
};

class TypeChecker : TypeCheckingEnv {
public:
  using TypeCheckingEnv::TypeCheckingEnv;

  void checkComponent(ComponentOp comp) {
    LLVM_DEBUG(llvm::dbgs() << "\n"; loggingScope = comp.getName();
               logLine() << "Checking component " << comp.getName() << ":\n"; ident++);
    ComponentScope scope(comp, getTypeBindings());
    for (auto &op : comp.getRegion().getOps()) {
      (void)typeCheckOp(&op, scope);
    }
    LLVM_DEBUG(ident--; logLine() << "Finished checking component " << comp.getName() << "\n";
               loggingScope = "");
  }

private:
  void markColumn(TypeBinding &binding) {
    assert(binding.getSlot());
    mlir::cast<ComponentSlot>(binding.getSlot())->markColumn();
  }

  /// If the binding has a slot and is either; a subclass of NondetReg, a subclass of a covariant
  /// Array of NondetReg, or has members of NondetReg type. Then mark the slot as a column in the
  /// constraint system.
  void
  markPotentialColumn(TypeBinding &binding, llvm::function_ref<InFlightDiagnostic()> emitError) {
    if (!binding.getSlot()) {
      return;
    }
    auto NondetReg = getTypeBindings().MaybeGet("NondetReg");
    if (failed(NondetReg)) {
      return;
    }
    auto NondetExtReg = getTypeBindings().MaybeGet("NondetExtReg");
    if (failed(NondetExtReg)) {
      return;
    }

    auto validSubtype = [&NondetReg, &NondetExtReg](const TypeBinding &Binding) {
      return succeeded(Binding.subtypeOf(*NondetReg)) ||
             succeeded(Binding.subtypeOf(*NondetExtReg));
    };

    if (validSubtype(binding)) {
      markColumn(binding);
    }

    if (binding.isArray()) {
      auto elt = binding.getArrayElement(emitError);
      assert(succeeded(elt));
      if (validSubtype(*elt)) {
        markColumn(binding);
      }
    }

    const auto *b = &binding;
    while (b) {
      b = [&]() -> const TypeBinding * {
        for (auto &member : b->getMembers()) {
          if (member.getValue().has_value() && validSubtype(*member.getValue())) {
            markColumn(binding);
            return nullptr;
          }
        }
        if (b->hasSuperType()) {
          return &b->getSuperType();
        }
        return nullptr;
      }();
    }
  }

  FailureOr<TypeBinding> typeCheckOp(Operation *op, Scope &scope) {
    assert(op != nullptr);
    LLVM_DEBUG(logLine() << "Checking " << op->getName() << "\n";
               if (op->getNumRegions() == 0) { logLine() << "  " << *op << "\n"; });
    if (getBindings().contains(op)) {
      auto &cachedBinding = getBindings().get(op);
      LLVM_DEBUG(logLine() << "Pulled from cache: "; if (failed(cachedBinding)) {
        llvm::dbgs() << "<<FAILURE>>\n";
      } else { llvm::dbgs() << *cachedBinding << "\n"; });
      return cachedBinding;
    }
    LLVM_DEBUG(ident++);
    auto operands = getOperandTypes(op, scope);
    LLVM_DEBUG(ident--);
    if (failed(operands)) {
      LLVM_DEBUG(logLine() << "Failed to obtain bindings of the operation's operands\n");
      return failure();
    }
    for (auto &rule : getRules()) {
      if (failed(rule->match(op))) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
      auto frame = rule->allocate(scope.getCurrentFrame());
      FailureOr<TypeBinding> ruleResult =
          succeeded(frame) ? applyRuleWithFrame(*frame, op, scope, *rule, *operands)
                           : applyRule(op, scope, *rule, *operands);

      LLVM_DEBUG(llvm::dbgs() << "\n");
      if (succeeded(ruleResult)) {
        LLVM_DEBUG(logLine() << "SUCCESS: Storing binding "; ruleResult->print(llvm::dbgs(), true);
                   llvm::dbgs() << "\n");
        markPotentialColumn(*ruleResult, [op] { return op->emitError(); });
        return getBindings().add(op, ruleResult);
      }
    }
    LLVM_DEBUG(logLine() << " FAILURE: Not rule was matched\n");
    return getBindings().add(
        op, op->emitError() << "could not deduce the type of op '" << op->getName() << "'"
    );
  }

  FailureOr<std::vector<TypeBinding>> getOperandTypes(Operation *op, Scope &scope) {
    auto operands = op->getOperands();
    std::vector<TypeBinding> operandBindings;
    for (auto operand : operands) {
      auto r = typeCheckValue(operand, scope);
      if (failed(r)) {
        return failure();
      }
      operandBindings.push_back(*r);
    }
    return operandBindings;
  }

  FailureOr<TypeBinding> typeCheckValue(Value v, Scope &scope) {
    auto op = v.getDefiningOp();
    if (op) {
      return typeCheckOp(op, scope);
    } else {
      if (getBindings().containsValue(v)) {
        return getBindings().getValue(v);
      }
      return failure();
    }
  }

  FailureOr<TypeBinding>
  applyRule(Operation *op, Scope &scope, const TypingRule &rule, ArrayRef<TypeBinding> operands) {
    // Check if the operation has regions and evaluate them first
    auto regions = op->getRegions();
    BlockScopesGuard regionScopes(scope, getTypeBindings());

    if (!std::all_of(regions.begin(), regions.end(), [&](auto &region) {
      auto valueBindings =
          rule.bindRegionArguments(ValueRange(region.getArguments()), op, operands, scope);
      if (failed(valueBindings)) {
        return false;
      }
      for (auto [value, type] : llvm::zip_equal(
               mlir::iterator_range(region.args_begin(), region.args_end()), *valueBindings
           )) {
        (void)getBindings().addValue(value, type);
      }
      return succeeded(typeCheckRegion(region, regionScopes.get()));
    })) {
      return failure();
    }

    return rule.typeCheck(op, operands, scope, regionScopes);
  }
  /// Handles the creation of the frame and linking it to the generated type binding if it
  /// succeeded.
  FailureOr<TypeBinding> applyRuleWithFrame(
      Frame frame, Operation *op, Scope &scope, const TypingRule &rule,
      ArrayRef<TypeBinding> operands
  ) {
    FrameScope frameScope(scope, frame);
    // Typecheck normally using a scope with the frame
    auto result = applyRule(op, frameScope, rule, operands);

    if (failed(result)) {
      return failure();
    }
    auto finalBinding = TypeBinding::ReplaceFrame(*result, frame);
    auto *parent = frame.getParentSlot();
    if (auto *compParent = mlir::dyn_cast_if_present<ComponentSlot>(parent)) {
      if (mlir::isa_and_present<ComponentSlot>(finalBinding.getSlot())) {
        // Remove the slot if it is a ComponentSlot since we are
        // going to store that component in the parent slot instead
        finalBinding.markSlot(nullptr);
      }
      compParent->setBinding(finalBinding);
    }
    return finalBinding;
  }

  LogicalResult typeCheckRegion(Region &region, Scope &scope) {
    bool anyFailed = false;

    for (auto &op : region.getOps()) {
      auto result = typeCheckOp(&op, scope);
      anyFailed = anyFailed || failed(result);
    }

    return anyFailed ? failure() : success();
  }
};

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
  root->walk([&](ComponentOp op) { indegrees[op.getName()]; });
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
      queue.push_back(compOp);
    }
  }

  // Kahn's algorithm will find the topological order or fail if there is a cycle.
  while (!queue.empty()) {
    auto op = queue.front();
    queue.erase(queue.begin());
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

} // namespace

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
  TypeChecker tc(*bindings, rules, typeBindings);
  for (auto op : sortedComponents) {
    tc.checkComponent(op);
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
      DirectiveTypeRule, LookupTypeRule, SwitchTypeRule, ConstructGlobalTypingRule>(bindings);

  return rules;
}

TypingRule::TypingRule(TypeBindings &Bindings) : bindings(&Bindings) {}
TypeBindings &TypingRule::getBindings() const { return *bindings; }
FrozenTypingRuleSet::FrozenTypingRuleSet(RuleSet &&Rules) : rules(std::move(Rules)) {}
FrozenTypingRuleSet::RuleSet::const_iterator FrozenTypingRuleSet::begin() const {
  return rules.begin();
}
FrozenTypingRuleSet::RuleSet::const_iterator FrozenTypingRuleSet::end() const {
  return rules.end();
}
TypingRuleSet::operator FrozenTypingRuleSet() { return FrozenTypingRuleSet(std::move(rules)); }
} // namespace zhl
