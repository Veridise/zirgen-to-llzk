#pragma once

#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"

namespace zhl {

class Scope {
public:
  virtual ~Scope() = default;

  // TODO: Global declarations
  virtual void declareGenericParam(mlir::StringRef, uint64_t, TypeBinding) = 0;
  virtual void declareConstructorParam(mlir::StringRef, uint64_t, TypeBinding) = 0;
  virtual void declareMember(mlir::StringRef) = 0;
  virtual void declareMember(mlir::StringRef, TypeBinding) = 0;
  virtual bool memberDeclaredWithType(mlir::StringRef) = 0;
  virtual void declareSuperType(TypeBinding) = 0;
  virtual zirgen::Zhl::ComponentOp getOp() const = 0;
  virtual mlir::FailureOr<TypeBinding> getSuperType() const = 0;
  virtual TypeBindings &getTypeBindings() = 0;
};

class ComponentScope : public Scope {
public:
  ComponentScope(zirgen::Zhl::ComponentOp component, TypeBindings &bindings);
  ~ComponentScope() override;

  // TODO: Add checks for duplicated names of parameters

  void declareGenericParam(mlir::StringRef name, uint64_t index, TypeBinding type) override;

  void declareConstructorParam(mlir::StringRef name, uint64_t index, TypeBinding type) override;

  void declareMember(mlir::StringRef name) override;

  void declareSuperType(TypeBinding type) override;

  /// Allows overriding a member if the current value is None
  void declareMember(mlir::StringRef name, TypeBinding type) override;

  bool memberDeclaredWithType(mlir::StringRef name) override;

  zirgen::Zhl::ComponentOp getOp() const override;

  mlir::FailureOr<TypeBinding> getSuperType() const override;

  TypeBindings &getTypeBindings() override;

private:
  TypeBindings *bindings;
  zirgen::Zhl::ComponentOp component;
  ParamsMap constructorParams;
  ParamsMap genericParams;
  MembersMap members;
  mlir::FailureOr<TypeBinding> superType;
};

class BlockScope : public Scope {
public:
  explicit BlockScope(Scope &, mlir::Region &);
  ~BlockScope() override;

  void declareGenericParam(mlir::StringRef name, uint64_t index, TypeBinding type) override;

  void declareConstructorParam(mlir::StringRef name, uint64_t index, TypeBinding type) override;

  void declareMember(mlir::StringRef name) override;

  void declareSuperType(TypeBinding type) override;

  /// Allows overriding a member if the current value is None
  void declareMember(mlir::StringRef name, TypeBinding type) override;

  bool memberDeclaredWithType(mlir::StringRef name) override;

  zirgen::Zhl::ComponentOp getOp() const override;

  mlir::FailureOr<TypeBinding> getSuperType() const override;

  TypeBindings &getTypeBindings() override;

private:
  Scope *parent;
  mlir::Region &region;
  MembersMap shadowedMembers;
  mlir::FailureOr<TypeBinding> superType;
};

} // namespace zhl
