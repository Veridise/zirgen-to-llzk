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

class Scope {
public:
  Scope(ComponentOp component, TypeBindings &bindings)
      : bindings(&bindings), component(component) {}
  ~Scope() {
    // TODO Commit all the stuff generated during type checking to the type bindings
    assert(superType.has_value());
    bindings->Create(component.getName(), *superType);
  }

  // TODO: Add checks for duplicated names of parameters

  void declareGenericParam(mlir::StringRef name, uint64_t index, TypeBinding type) {
    genericParams.insert({{name, index}, type});
  }

  void declareConstructorParam(mlir::StringRef name, uint64_t index, TypeBinding type) {
    constructorParams.insert({{name, index}, type});
  }

  void declareMember(mlir::StringRef name) {
    assert(members.find(name) == members.end());
    members[name] = std::nullopt;
  }

  void declareSuperType(TypeBinding type) { superType = type; }

  /// Allows overriding a member if the current value is None
  void declareMember(mlir::StringRef name, TypeBinding type) {
    if (members.find(name) != members.end()) {
      // If the value is present and we are (re-)declaring it
      // we can only do so if it has not value.
      assert(!members[name].has_value());
    }
    members[name] = type;
  }

  bool memberDeclaredWithType(mlir::StringRef name) { return members[name].has_value(); }

  ComponentOp getOp() const { return component; }

private:
  TypeBindings *bindings;
  ComponentOp component;
  std::map<std::pair<std::string_view, uint64_t>, TypeBinding> constructorParams;
  std::map<std::string_view, std::optional<TypeBinding>> members;
  std::map<std::pair<std::string_view, uint64_t>, TypeBinding> genericParams;
  std::optional<TypeBinding> superType;
};

class TypingRule {
public:
  explicit TypingRule(const TypeBindings &bindings) : bindings(&bindings) {}

  const TypeBindings &getBindings() const { return *bindings; }
  virtual mlir::FailureOr<TypeBinding>
  typeCheck(mlir::Operation *, mlir::ArrayRef<TypeBinding>, Scope &) const = 0;

private:
  const TypeBindings *bindings;
};

class OpBindings {
public:
  virtual ~OpBindings() = default;
  virtual mlir::FailureOr<TypeBinding> add(mlir::Operation *, mlir::FailureOr<TypeBinding>) = 0;
  virtual const mlir::FailureOr<TypeBinding> &get(mlir::Operation *) const = 0;
  virtual bool contains(mlir::Operation *) const = 0;
  void dump() const { print(llvm::dbgs()); }
  virtual void print(llvm::raw_ostream &) const = 0;
  virtual bool missingBindings() const = 0;
  operator LogicalResult() const { return missingBindings() ? mlir::failure() : mlir::success(); }

protected:
  void printBinding(llvm::raw_ostream &os, mlir::FailureOr<TypeBinding> type) const {
    os << " : ";
    if (mlir::succeeded(type)) {
      type->print(os);
    } else {
      os << "<type checking failed>";
    }
    os << "\n";
  }
};

template <typename Key> class OpBindingsMap : public OpBindings {
public:
  using map = mlir::DenseMap<Key, mlir::FailureOr<TypeBinding>>;

  typename map::iterator begin() { return bindings.begin(); }
  typename map::const_iterator begin() const { return bindings.begin(); }
  typename map::iterator end() { return bindings.end(); }
  typename map::const_iterator end() const { return bindings.end(); }

  bool contains(mlir::Operation *op) const final { return containsKey(opToKey(op)); }

  bool missingBindings() const override {
    return std::any_of(begin(), end(), [](auto &pair) { return mlir::failed(pair.second); });
  }
  void print(llvm::raw_ostream &os) const final {
    for (auto &[op, type] : bindings) {
      printKey(op, os);
      printBinding(os, type);
    }
  }
  mlir::FailureOr<TypeBinding> add(mlir::Operation *op, mlir::FailureOr<TypeBinding> type) final {
    validateOp(op);
    return addType(opToKey(op), type);
  }
  const mlir::FailureOr<TypeBinding> &get(mlir::Operation *op) const final {
    validateOp(op);
    return getType(opToKey(op));
  }

  mlir::FailureOr<TypeBinding> addType(Key k, mlir::FailureOr<TypeBinding> type) {
    bindings.insert({k, type});
    return type;
  }

  const mlir::FailureOr<TypeBinding> &getType(Key k) const {
    if (bindings.find(k) == end()) {
      return failed;
    }
    return bindings.at(k);
  }

  bool containsKey(Key k) const { return bindings.find(k) != end(); }

protected:
  virtual Key opToKey(mlir::Operation *) const = 0;
  virtual void validateOp(mlir::Operation *) const = 0;
  virtual void printKey(Key k, llvm::raw_ostream &os) const = 0;

private:
  mlir::FailureOr<TypeBinding> failed;
  map bindings;
};

class ValueBindings : public OpBindingsMap<mlir::Value> {
protected:
  mlir::Value opToKey(mlir::Operation *op) const final { return op->getResult(0); }

  void validateOp(mlir::Operation *op) const final {
    assert(op->getNumResults() == 1 && "expected an op with only 1 result");
  }
  void printKey(mlir::Value k, llvm::raw_ostream &os) const final { os << k; }
};

class StmtBindings : public OpBindingsMap<mlir::Operation *> {
protected:
  mlir::Operation *opToKey(mlir::Operation *op) const final { return op; }

  void validateOp(mlir::Operation *op) const final {
    assert(op->getNumResults() == 0 && "expected an op with no results");
  }
  void printKey(mlir::Operation *k, llvm::raw_ostream &os) const final { os << *k; }
};

class ZhlOpBindings : public OpBindings {
public:
  mlir::FailureOr<TypeBinding> addValue(mlir::Value v, mlir::FailureOr<TypeBinding> type) {
    return values.addType(v, type);
  }

  mlir::FailureOr<TypeBinding> add(mlir::Operation *op, mlir::FailureOr<TypeBinding> type) final {
    if (opIsValue(op)) {
      return values.add(op, type);
    } else {
      return stmts.add(op, type);
    }
  }

  const mlir::FailureOr<TypeBinding> &getValue(mlir::Value v) const { return values.getType(v); }

  const mlir::FailureOr<TypeBinding> &get(mlir::Operation *op) const final {
    return opIsValue(op) ? values.get(op) : stmts.get(op);
  }

  bool contains(mlir::Operation *op) const final {
    return (opIsValue(op) && values.contains(op)) || stmts.contains(op);
  }
  bool containsValue(mlir::Value v) const { return values.containsKey(v); }
  void print(llvm::raw_ostream &os) const final {
    values.print(os);
    stmts.print(os);
  }
  bool missingBindings() const final { return values.missingBindings() || stmts.missingBindings(); }

private:
  bool opIsValue(mlir::Operation *op) const { return op->getNumResults() == 1; }

  ValueBindings values;
  StmtBindings stmts;
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

  std::unique_ptr<OpBindings> check(mlir::Operation *root, TypeBindings &typeBindings) {
    mlir::SymbolTable st(root);
    std::unique_ptr<ZhlOpBindings> bindings = std::make_unique<ZhlOpBindings>();
    root->walk([&](ComponentOp op) { checkComponent(op, *bindings, typeBindings); });
    return bindings;
  }

private:
  void checkComponent(ComponentOp comp, ZhlOpBindings &bindings, TypeBindings &typeBindings) {
    Scope scope(comp, typeBindings);
    comp.walk([&](mlir::Operation *op) {
      if (mlir::isa<ComponentOp>(op)) {
        return;
      }
      (void)typeCheckOp(op, bindings, scope);
    });
  }

  mlir::FailureOr<TypeBinding>
  typeCheckValue(mlir::Value v, ZhlOpBindings &bindings, Scope &scope) {
    auto op = v.getDefiningOp();
    if (op) {
      return typeCheckOp(op, bindings, scope);
    } else {
      if (bindings.containsValue(v)) {
        return bindings.getValue(v);
      }
      return mlir::failure();
    }
  }

  mlir::FailureOr<TypeBinding>
  typeCheckOp(mlir::Operation *op, ZhlOpBindings &bindings, Scope &scope) {
    assert(op != nullptr);
    if (bindings.contains(op)) {
      return bindings.get(op);
    }
    auto operands = getOperandTypes(op, bindings, scope);
    if (mlir::failed(operands)) {
      return mlir::failure();
    }
    for (auto &rule : rules) {
      auto checkResult = rule->typeCheck(op, *operands, scope);
      if (mlir::succeeded(checkResult)) {
        return bindings.add(op, checkResult);
      }
    }
    return bindings.add(
        op,
        op->emitOpError() << "could not deduce the type of requested op '" << op->getName() << "'"
    );
  }

  mlir::FailureOr<std::vector<TypeBinding>>
  getOperandTypes(mlir::Operation *op, ZhlOpBindings &bindings, Scope &scope) {
    auto operands = op->getOperands();
    std::vector<TypeBinding> operandBindings;
    for (auto operand : operands) {
      auto r = typeCheckValue(operand, bindings, scope);
      if (mlir::failed(r)) {
        return mlir::failure();
      }
      operandBindings.push_back(*r);
    }
    return operandBindings;
  }

  std::vector<std::unique_ptr<TypingRule>> rules;
};

template <typename Op> class OpTypingRule : public TypingRule {
public:
  using TypingRule::TypingRule;

  mlir::FailureOr<TypeBinding> typeCheck(
      mlir::Operation *op, mlir::ArrayRef<TypeBinding> operands, Scope &scope
  ) const override {
    assert(op->getParentOfType<ComponentOp>() == scope.getOp());
    if (auto o = mlir::dyn_cast<Op>(op)) {
      return typeCheck(o, operands, scope);
    }
    return mlir::failure();
  }

  virtual mlir::FailureOr<TypeBinding>
  typeCheck(Op op, mlir::ArrayRef<TypeBinding> operands, Scope &scope) const = 0;
};

class LiteralTypingRule : public OpTypingRule<LiteralOp> {
public:
  using OpTypingRule<LiteralOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(LiteralOp op, mlir::ArrayRef<TypeBinding>, Scope &scope) const override {
    return getBindings().Get("Val");
  }
};

class StringTypingRule : public OpTypingRule<StringOp> {
public:
  using OpTypingRule<StringOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(StringOp op, mlir::ArrayRef<TypeBinding>, Scope &scope) const override {
    return getBindings().Get("String");
  }
};

class GlobalTypingRule : public OpTypingRule<GlobalOp> {
public:
  using OpTypingRule<GlobalOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(GlobalOp op, mlir::ArrayRef<TypeBinding>, Scope &scope) const override {
    auto binding = getBindings().MaybeGet(op.getName());
    if (mlir::failed(binding)) {
      return op->emitError() << "type '" << op.getName() << "' was not found";
    }
    return binding;
  }
};

class ParameterTypingRule : public OpTypingRule<ConstructorParamOp> {
public:
  using OpTypingRule<ConstructorParamOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding> typeCheck(
      ConstructorParamOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope
  ) const override {
    // TODO: Annotation of the component being analyzed with the parameter name and index
    if (operands.empty()) {
      return mlir::failure();
    }
    return op.getVariadic() ? TypeBinding::WrapVariadic(operands[0]) : operands[0];
  }
};

class ExternTypingRule : public OpTypingRule<ExternOp> {
public:
  using OpTypingRule<ExternOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(ExternOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope) const override {
    if (operands.empty()) {
      return mlir::failure();
    }
    return operands[0];
  }
};

class ConstructTypingRule : public OpTypingRule<ConstructOp> {
public:
  using OpTypingRule<ConstructOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(ConstructOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope) const override {
    if (operands.empty()) {
      return mlir::failure();
    }
    return operands[0];
  }
};

class GetGlobalTypingRule : public OpTypingRule<GetGlobalOp> {
public:
  using OpTypingRule<GetGlobalOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(GetGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope) const override {
    // TODO: Add check that the global exists in the scope
    if (operands.empty()) {
      return mlir::failure();
    }
    return operands[0];
  }
};

class ConstructGlobalTypingRule : public OpTypingRule<ConstructGlobalOp> {
public:
  using OpTypingRule<ConstructGlobalOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding> typeCheck(
      ConstructGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope
  ) const override {
    // TODO: Add global declaration to the scope
    if (operands.empty()) {
      return mlir::failure();
    }
    return operands[0];
  }
};

class SuperTypingRule : public OpTypingRule<SuperOp> {
public:
  using OpTypingRule<SuperOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(SuperOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope) const override {
    if (operands.empty()) {
      return mlir::failure();
    }
    scope.declareSuperType(operands[0]);
    return operands[0];
  }
};

class DeclareTypingRule : public OpTypingRule<DeclarationOp> {
public:
  using OpTypingRule<DeclarationOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(DeclarationOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope) const override {
    if (operands.empty()) {
      scope.declareMember(op.getMember());
      return getBindings().Bottom();
    } else {
      scope.declareMember(op.getMember(), operands[0]);
      return operands[0];
    }
  }
};

class DefineTypeRule : public OpTypingRule<DefinitionOp> {
public:
  using OpTypingRule<DefinitionOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(DefinitionOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope) const override {
    auto decl = getDeclaration(op);
    if (mlir::failed(decl)) {
      return op->emitError() << "Malformed IR: Definition must have a declaration operand";
    }
    if (operands.size() < 2) {
      return mlir::failure();
    }
    scope.declareMember(decl->getMember(), operands[1]);
    return operands[1];
  }

private:
  mlir::FailureOr<DeclarationOp> getDeclaration(DefinitionOp op) const {
    auto decl = op.getDeclaration().getDefiningOp();
    if (!decl) {
      return mlir::failure();
    }
    if (auto declOp = mlir::dyn_cast<DeclarationOp>(decl)) {
      return declOp;
    }
    return mlir::failure();
  }
};

class ConstrainTypeRule : public OpTypingRule<ConstraintOp> {
public:
  using OpTypingRule<ConstraintOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(ConstraintOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope) const override {
    // TODO: Check that the argument types are correct
    return getBindings().Component();
  }
};

namespace {

class PrintTypeBindingsPass : public PrintTypeBindingsBase<PrintTypeBindingsPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    TypeBindings bindings;
    zkc::Zmir::addBuiltinBindings(bindings);

    TypingRuleSet rules;
    rules.add<
        LiteralTypingRule, StringTypingRule, GlobalTypingRule, ParameterTypingRule, SuperTypingRule,
        GetGlobalTypingRule, ConstructTypingRule, ExternTypingRule, DeclareTypingRule,
        ConstrainTypeRule, DefineTypeRule>(bindings);

    auto result = rules.check(mod, bindings);
    result->dump();
    if (mlir::failed(*result)) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPrintTypeBindingsPass() {
  return std::make_unique<PrintTypeBindingsPass>();
}

} // namespace zhl
