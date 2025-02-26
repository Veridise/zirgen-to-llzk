#pragma once

#include <zklang/Dialect/ZHL/Typing/Scope.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZHL/Typing/Typing.h>

namespace zhl {

class LiteralTypingRule : public OpTypingRule<zirgen::Zhl::LiteralOp> {
public:
  using OpTypingRule<zirgen::Zhl::LiteralOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::LiteralOp op, mlir::ArrayRef<TypeBinding>, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class StringTypingRule : public OpTypingRule<zirgen::Zhl::StringOp> {
public:
  using OpTypingRule<zirgen::Zhl::StringOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::StringOp op, mlir::ArrayRef<TypeBinding>, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class GlobalTypingRule : public OpTypingRule<zirgen::Zhl::GlobalOp> {
public:
  using OpTypingRule<zirgen::Zhl::GlobalOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::GlobalOp op, mlir::ArrayRef<TypeBinding>, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class ParameterTypingRule : public OpTypingRule<zirgen::Zhl::ConstructorParamOp> {
public:
  using OpTypingRule<zirgen::Zhl::ConstructorParamOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::ConstructorParamOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class ExternTypingRule : public OpTypingRule<zirgen::Zhl::ExternOp> {
public:
  using OpTypingRule<zirgen::Zhl::ExternOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::ExternOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class ConstructTypingRule : public OpTypingRule<zirgen::Zhl::ConstructOp> {
public:
  using OpTypingRule<zirgen::Zhl::ConstructOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::ConstructOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class GetGlobalTypingRule : public OpTypingRule<zirgen::Zhl::GetGlobalOp> {
public:
  using OpTypingRule<zirgen::Zhl::GetGlobalOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::GetGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class ConstructGlobalTypingRule : public OpTypingRule<zirgen::Zhl::ConstructGlobalOp> {
public:
  using OpTypingRule<zirgen::Zhl::ConstructGlobalOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::ConstructGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class SuperTypingRule : public OpTypingRule<zirgen::Zhl::SuperOp> {
public:
  using OpTypingRule<zirgen::Zhl::SuperOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::SuperOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class DeclareTypingRule : public OpTypingRule<zirgen::Zhl::DeclarationOp> {
public:
  using OpTypingRule<zirgen::Zhl::DeclarationOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::DeclarationOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class DefineTypeRule : public OpTypingRule<zirgen::Zhl::DefinitionOp> {
public:
  using OpTypingRule<zirgen::Zhl::DefinitionOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::DefinitionOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class ConstrainTypeRule : public OpTypingRule<zirgen::Zhl::ConstraintOp> {
public:
  using OpTypingRule<zirgen::Zhl::ConstraintOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::ConstraintOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class GenericParamTypeRule : public OpTypingRule<zirgen::Zhl::TypeParamOp> {
public:
  using OpTypingRule<zirgen::Zhl::TypeParamOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::TypeParamOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class SpecializeTypeRule : public OpTypingRule<zirgen::Zhl::SpecializeOp> {
public:
  using OpTypingRule<zirgen::Zhl::SpecializeOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::SpecializeOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class SubscriptTypeRule : public OpTypingRule<zirgen::Zhl::SubscriptOp> {
public:
  using OpTypingRule<zirgen::Zhl::SubscriptOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::SubscriptOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class ArrayTypeRule : public OpTypingRule<zirgen::Zhl::ArrayOp> {
public:
  using OpTypingRule<zirgen::Zhl::ArrayOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::ArrayOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class BackTypeRule : public OpTypingRule<zirgen::Zhl::BackOp> {
public:
  using OpTypingRule<zirgen::Zhl::BackOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::BackOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class RangeTypeRule : public OpTypingRule<zirgen::Zhl::RangeOp> {
public:
  using OpTypingRule<zirgen::Zhl::RangeOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::RangeOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class ReduceTypeRule : public OpTypingRule<zirgen::Zhl::ReduceOp> {
public:
  using OpTypingRule<zirgen::Zhl::ReduceOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::ReduceOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;

  mlir::FailureOr<Frame> allocate(Frame) const override;
};

class ConstructGlobalTypeRule : public OpTypingRule<zirgen::Zhl::ConstructGlobalOp> {
public:
  using OpTypingRule<zirgen::Zhl::ConstructGlobalOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::ConstructGlobalOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;
};

class BlockTypeRule : public OpTypingRule<zirgen::Zhl::BlockOp> {
public:
  using OpTypingRule<zirgen::Zhl::BlockOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::BlockOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;

  mlir::FailureOr<Frame> allocate(Frame) const override;
};

class SwitchTypeRule : public OpTypingRule<zirgen::Zhl::SwitchOp> {
public:
  using OpTypingRule<zirgen::Zhl::SwitchOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::SwitchOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;

  mlir::FailureOr<Frame> allocate(Frame) const override;
};

class MapTypeRule : public OpTypingRule<zirgen::Zhl::MapOp> {
public:
  using OpTypingRule<zirgen::Zhl::MapOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::MapOp op, mlir::ArrayRef<TypeBinding> operands, Scope &scope, mlir::ArrayRef<const Scope *>)
      const override;

  mlir::FailureOr<std::vector<TypeBinding>>
  bindRegionArguments(mlir::ValueRange args, zirgen::Zhl::MapOp, mlir::ArrayRef<TypeBinding>, Scope &)
      const override;

  mlir::FailureOr<Frame> allocate(Frame) const override;
};

class LookupTypeRule : public OpTypingRule<zirgen::Zhl::LookupOp> {
public:
  using OpTypingRule<zirgen::Zhl::LookupOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::LookupOp, mlir::ArrayRef<TypeBinding>, Scope &, mlir::ArrayRef<const Scope *>)
      const override;
};

class DirectiveTypeRule : public OpTypingRule<zirgen::Zhl::DirectiveOp> {
public:
  using OpTypingRule<zirgen::Zhl::DirectiveOp>::OpTypingRule;

  mlir::FailureOr<TypeBinding>
  typeCheck(zirgen::Zhl::DirectiveOp, mlir::ArrayRef<TypeBinding>, Scope &, mlir::ArrayRef<const Scope *>)
      const override;
};

} // namespace zhl
