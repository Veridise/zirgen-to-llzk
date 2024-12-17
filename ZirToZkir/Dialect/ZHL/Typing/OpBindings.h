#pragma once

#include "ZirToZkir/Dialect/ZHL/Typing/TypeBindings.h"
#include <llvm/Support/Debug.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace zhl {

class OpBindings {
public:
  virtual ~OpBindings() = default;
  virtual mlir::FailureOr<TypeBinding> add(mlir::Operation *, mlir::FailureOr<TypeBinding>) = 0;
  virtual const mlir::FailureOr<TypeBinding> &get(mlir::Operation *) const = 0;
  virtual bool contains(mlir::Operation *) const = 0;
  virtual void print(llvm::raw_ostream &) const = 0;
  virtual bool missingBindings() const = 0;

  operator mlir::LogicalResult() const;
  void dump() const;

protected:
  void printBinding(llvm::raw_ostream &os, mlir::FailureOr<TypeBinding> type) const;
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
  mlir::Value opToKey(mlir::Operation *op) const final;
  void validateOp(mlir::Operation *op) const final;
  void printKey(mlir::Value k, llvm::raw_ostream &os) const final;
};

class StmtBindings : public OpBindingsMap<mlir::Operation *> {
protected:
  mlir::Operation *opToKey(mlir::Operation *op) const final;
  void validateOp(mlir::Operation *op) const final;
  void printKey(mlir::Operation *k, llvm::raw_ostream &os) const final;
};

class ZhlOpBindings : public OpBindings {
public:
  mlir::FailureOr<TypeBinding> addValue(mlir::Value v, mlir::FailureOr<TypeBinding> type);

  mlir::FailureOr<TypeBinding> add(mlir::Operation *op, mlir::FailureOr<TypeBinding> type) final;

  const mlir::FailureOr<TypeBinding> &getValue(mlir::Value v) const;

  const mlir::FailureOr<TypeBinding> &get(mlir::Operation *op) const final;

  bool contains(mlir::Operation *op) const final;
  bool containsValue(mlir::Value v) const;
  void print(llvm::raw_ostream &os) const final;
  bool missingBindings() const final;

private:
  bool opIsValue(mlir::Operation *op) const;

  ValueBindings values;
  StmtBindings stmts;
};

} // namespace zhl
