//===- Patterns.cpp - ZHL->LLZK struct conversion patterns ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the available conversion patterns for converting ZHL
// component operations into LLZK struct and function operations.
//
//===----------------------------------------------------------------------===//

#include <mlir/IR/BuiltinOps.h>
#include <zklang/Passes/ConvertZhlToLlzkStruct/Patterns.h>

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <llzk/Dialect/Function/IR/Dialect.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Shared/OpHelpers.h>
#include <llzk/Dialect/Struct/IR/Dialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/LLZK/Builder.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
#include <zklang/Dialect/ZHL/ZhlOpConversionPattern.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>
#include <zklang/Dialect/ZML/Utils/Helpers.h>

#define DEBUG_TYPE "convert-zhl-to-llzk-struct"

using namespace zklang;
using namespace mlir;
using namespace zirgen::Zhl;
using namespace llzk::component;
using namespace llzk::function;
using namespace zhl;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool parentIsSelf(Operation *op) {
  return op->getParentOp() && mlir::isa<zml::SelfOp>(op->getParentOp());
}

//===----------------------------------------------------------------------===//
// LowerComponentToLlzkStruct
//===----------------------------------------------------------------------===//

class LowerComponentToLlzkStruct : public ZhlOpConversionPattern<ComponentOp> {
public:
  using ZhlOpConversionPattern<ComponentOp>::ZhlOpConversionPattern;

  /// Match only components that have the type binding attribute and that are not externs
  LogicalResult match(ComponentOp, Binding typeBinding, BindingsAdaptor) const override {
    return failure(typeBinding->isExtern());
  }

  Location findSuperLocation(ComponentOp op) const {
    auto &blocks = op.getRegion().getBlocks();
    if (!blocks.empty()) {

      auto term = blocks.front().getTerminator();
      if (term) {
        return term->getLoc();
      }
    }
    return op.getLoc();
  }

  void rewrite(
      ComponentOp op, Binding typeBinding, OpAdaptor, BindingsAdaptor,
      ConversionPatternRewriter &rewriter
  ) const override {
    auto *tc = getTypeConverter();
    assert(tc);
    llzk::ComponentBuilder builder;
    auto genericNames = typeBinding->getGenericParamNames();
    auto paramLocations = typeBinding->getConstructorParamLocations();

    auto super = typeBinding.getSuperTypeAttr().materializeType(tc);
    auto superLoc = findSuperLocation(op);

    builder.name(typeBinding->getName())
        .location(op->getLoc())
        .attrs(op->getAttrs())
        .typeParams(genericNames)
        .constructor(typeBinding.materializeCtorType(), paramLocations)
        .takeRegion(&op.getRegion())
        .field("$super", super, superLoc);
    auto comp = builder.build(rewriter, *tc);

    rewriter.replaceOp(op.getOperation(), comp.getOperation());
  }
};

//===----------------------------------------------------------------------===//
// LowerConstructorParamOpToLlzkStruct
//===----------------------------------------------------------------------===//

/// Replaces a zhl.parameter with a block argument from the function.
class LowerConstructorParamOpToLlzkStruct : public ZhlOpConversionPattern<ConstructorParamOp> {
public:
  using ZhlOpConversionPattern<ConstructorParamOp>::ZhlOpConversionPattern;

  LogicalResult match(ConstructorParamOp, Binding, BindingsAdaptor) const final {
    return success();
  }

  void rewrite(
      ConstructorParamOp op, Binding, OpAdaptor adaptor, BindingsAdaptor,
      ConversionPatternRewriter &rewriter
  ) const final {
    auto body = op->getParentOfType<llzk::function::FuncDefOp>();
    auto baseOffset = llvm::StringSwitch<unsigned>(body.getName())
                          .Case("compute", 0)
                          .Case("constrain", 1)
                          .Default(0);
    mlir::BlockArgument arg = body.getArgument(baseOffset + adaptor.getIndex());

    rewriter.replaceOp(op, arg);
  }
};

//===----------------------------------------------------------------------===//
// LowerExternComponentToLlzkFunction
//===----------------------------------------------------------------------===//

class LowerExternComponentToLlzkFunction : public ZhlOpConversionPattern<ComponentOp> {
public:
  using ZhlOpConversionPattern<ComponentOp>::ZhlOpConversionPattern;

  /// Match only components that have the type binding attribute and that are externs
  LogicalResult matchAndRewrite(
      ComponentOp op, Binding typeBinding, OpAdaptor, BindingsAdaptor,
      ConversionPatternRewriter &rewriter
  ) const override {
    if (!typeBinding->isExtern()) {
      return failure();
    }
    SmallVector<Type> inputs;
    auto *tc = getTypeConverter();
    assert(tc);
    auto ctor = typeBinding.materializeCtorType();
    if (failed(tc->convertTypes(ctor.getInputs(), inputs))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<FuncDefOp>(
        op, op.getName(),
        rewriter.getFunctionType(
            inputs, {tc->convertType(typeBinding.getSuperTypeAttr().materializeType())}
        ),
        ArrayRef({rewriter.getNamedAttr("sym_visibility", rewriter.getStringAttr("private"))}),
        ArrayRef<DictionaryAttr>({})
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerLookupOpToLlzkStruct
//===----------------------------------------------------------------------===//

/// Replaces a zhl.lookup with a field read operation.
class LowerLookupOpToLlzkStruct : public ZhlOpConversionPattern<LookupOp> {
public:
  using ZhlOpConversionPattern<LookupOp>::ZhlOpConversionPattern;

  Value maybeCastComponent(Value comp, Binding binding, OpBuilder &builder, Location loc) const {

    LLVM_DEBUG(llvm::dbgs() << "comp value: " << comp << '\n');
    LLVM_DEBUG(llvm::dbgs() << "type binding for the component: " << *binding << '\n';
               llvm::dbgs() << "Full printout: \n"; binding->print(llvm::dbgs(), true);
               llvm::dbgs() << '\n');
    auto type = binding.materializeType();
    LLVM_DEBUG(llvm::dbgs() << "     which materializes to " << type << '\n');
    if (comp.getType() != type) {
      LLVM_DEBUG(llvm::dbgs() << "Casting " << comp.getType() << " to " << type << '\n');
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, comp);
      return cast.getResult(0);
    }

    return comp;
  }

  FailureOr<Type> findDefiningType(LookupOp op, Binding it) const {
    LLVM_DEBUG(
        llvm::dbgs() << "Starting search for member " << op.getMember() << " with type " << *it
                     << '\n'
    );
    while (it && !it->definesMember(op.getMember(), /*recurse=*/false)) {
      it = it.getSuperTypeAttr();
      if (!it) {
        LLVM_DEBUG(llvm::dbgs() << "  Failed to get the super type\n");
        return op->emitError() << "member " << op.getMember() << " was not found";
      }
      LLVM_DEBUG(llvm::dbgs() << "Trying again with super type " << *it << '\n');
    }
    return it.materializeType();
  }

  FailureOr<Type> getField(LookupOp op, Binding binding) const {
    auto field = binding->getMember(op.getMember(), [&op] { return op->emitError(); });
    if (failed(field)) {
      return failure();
    }

    return zml::materializeTypeBinding(op.getContext(), *field);
  }

  Value
  createReadOps(Value comp, Type type, Type fieldType, LookupOp op, OpBuilder &builder) const {
    Value val = builder.create<FieldReadOp>(op.getLoc(), fieldType, comp, op.getMemberAttr());
    if (fieldType == type) {
      return val;
    }
    return builder.create<zml::SuperCoerceOp>(op.getLoc(), type, val);
  }

#define bail(x)                                                                                    \
  if (x) {                                                                                         \
    return failure();                                                                              \
  }

#define bail_failure(x) bail(failed(x))

  LogicalResult matchAndRewrite(
      LookupOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor,
      ConversionPatternRewriter &rewriter
  ) const final {
    auto field = getField(op, binding);
    bail_failure(field);

    auto defining = findDefiningType(op, binding);
    bail_failure(defining);

    auto comp = maybeCastComponent(adaptor.getComponent(), binding, rewriter, op.getLoc());
    if (*defining != comp.getType()) {
      // Coerce to the type in the chain that defines the accessed member
      comp = rewriter.create<zml::SuperCoerceOp>(op.getLoc(), *defining, comp);
    }

    auto replacement = createReadOps(comp, binding.materializeType(), *field, op, rewriter);
    rewriter.replaceOp(op, replacement);
    return mlir::success();
  }

#undef bail
#undef bail_failure
};

//===----------------------------------------------------------------------===//
// LowerDefinitionOpToLlzkStructInCompute
//===----------------------------------------------------------------------===//

/// Replaces a zhl.define with a field write.
class LowerDefinitionOpToLlzkStructInCompute : public ZhlOpConversionPattern<DefinitionOp> {
public:
  using ZhlOpConversionPattern<DefinitionOp>::ZhlOpConversionPattern;

  LogicalResult match(DefinitionOp op, Binding, BindingsAdaptor) const final {
    return success(zml::opIsInFunc("compute", op));
  }

  zhl::ComponentSlot *getSlot(Binding b) const {
    return dyn_cast_if_present<zhl::ComponentSlot>(b->getSlot());
  }

  zhl::ComponentSlot *findSlot(Binding opBinding, BindingsAdaptor bindings) const {
    if (auto *slot = getSlot(opBinding)) {
      return slot;
    }
    if (auto *slot = getSlot(bindings.getDeclaration())) {
      return slot;
    }

    return nullptr;
  }

  Type getSlotType(zhl::ComponentSlot *slot) const {
    return zml::materializeTypeBinding(getContext(), slot->getBinding());
  }

  void rewrite(
      DefinitionOp op, Binding opBinding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const final {
    auto *slot = findSlot(opBinding, bindings);

    // If the binding of this op has a slot then it is responsible of creating it.
    // Otherwise, check if the declaration's binding has a slot. If it does create it here.
    // If a slot is created here that means that the value we are storing in the field does not need
    // memory according to the type checker and thus we don't need to introduce the use-def cut
    // since that value is safe to use in @constrain functions.
    if (slot) {
      SmallVector<Operation *, 2> castOps;
      Value result = zml::CastHelper::getCastedValue(
          adaptor.getDefinition(), *bindings.getDefinition(), rewriter, castOps, getSlotType(slot)
      );

      zml::createAndStoreSlot(op, result, rewriter, *getTypeConverter(), slot);
    }

    rewriter.eraseOp(op);
  }
};

//===----------------------------------------------------------------------===//
// LowerDefinitionOpToLlzkStructInConstrain
//===----------------------------------------------------------------------===//

/// Removes a zhl.define.
class LowerDefinitionOpToLlzkStructInConstrain : public OpConversionPattern<DefinitionOp> {
public:
  using OpConversionPattern<DefinitionOp>::OpConversionPattern;

  LogicalResult match(DefinitionOp op) const final {
    return success(zml::opIsInFunc("constrain", op));
  }

  void rewrite(DefinitionOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};

//===----------------------------------------------------------------------===//
// LowerTopLevelSuperOpToLlzkStructInCompute
//===----------------------------------------------------------------------===//

// Top level means that the parent of SuperOp is SelfOp.
/// Replaces the top level zhl.super with a field write.
class LowerTopLevelSuperOpToLlzkStructInCompute : public ZhlOpConversionPattern<SuperOp> {
public:
  using ZhlOpConversionPattern<SuperOp>::ZhlOpConversionPattern;

  LogicalResult match(SuperOp op, Binding, BindingsAdaptor) const final {
    return success(zml::opIsInFunc("compute", op) && parentIsSelf(op));
  }

  void rewrite(
      SuperOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor,
      ConversionPatternRewriter &rewriter
  ) const final {
    auto *tc = getTypeConverter();
    assert(tc);

    Value value =
        *zml::CastHelper::getCastedValue(adaptor.getValue(), rewriter, binding.materializeType());
    auto finalType = tc->convertType(value.getType());
    if (finalType != value.getType()) {
      value =
          rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), finalType, value).getResult(0);
    }
    Value self = op->getParentOfType<zml::SelfOp>().getSelfValue();
    self =
        rewriter
            .create<UnrealizedConversionCastOp>(op.getLoc(), tc->convertType(self.getType()), self)

            .getResult(0);

    rewriter.replaceOpWithNewOp<FieldWriteOp>(op, self, "$super", value);
  }
};

//===----------------------------------------------------------------------===//
// LowerTopLevelSuperOpToLlzkStructInConstrain
//===----------------------------------------------------------------------===//

/// Removes the top level zhl.super.
class LowerTopLevelSuperOpToLlzkStructInConstrain : public OpConversionPattern<SuperOp> {
public:
  using OpConversionPattern<SuperOp>::OpConversionPattern;

  LogicalResult match(SuperOp op) const final {
    return success(zml::opIsInFunc("constrain", op) && parentIsSelf(op));
  }

  void rewrite(SuperOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};

} // namespace

void zklang::populateZhlToLlzkStructConversionPatterns(
    const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns
) {
  populateZhlComponentToLlzkStructConversionPatterns(tc, patterns);
  patterns.add<
      // clang-format off
      LowerConstructorParamOpToLlzkStruct,
      LowerDefinitionOpToLlzkStructInCompute,
      LowerDefinitionOpToLlzkStructInConstrain,
      LowerLookupOpToLlzkStruct,
      LowerTopLevelSuperOpToLlzkStructInCompute,
      LowerTopLevelSuperOpToLlzkStructInConstrain
      // clang-format on
      >(tc, patterns.getContext());
}

void zklang::populateZhlToLlzkStructConversionTarget(mlir::ConversionTarget &target) {
  populateZhlComponentToLlzkStructConversionTarget(target);
  target.addIllegalOp<LookupOp, DefinitionOp, ConstructorParamOp>();
  target.addDynamicallyLegalOp<SuperOp>([](SuperOp op) { return !parentIsSelf(op); });
}

void zklang::populateZhlToLlzkStructConversionPatternsAndLegality(
    const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target
) {
  populateZhlToLlzkStructConversionPatterns(tc, patterns);
  populateZhlToLlzkStructConversionTarget(target);
}

void zklang::populateZhlComponentToLlzkStructConversionPatterns(
    const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns
) {
  patterns.add<
      // clang-format off
      LowerComponentToLlzkStruct, 
      LowerExternComponentToLlzkFunction
      // clang-format on
      >(tc, patterns.getContext());
}

void zklang::populateZhlComponentToLlzkStructConversionTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<ZhlDialect, StructDialect, FunctionDialect, zml::ZMLDialect>();
  target.addIllegalOp<ComponentOp>();
  target.addLegalDialect<BuiltinDialect>();
}

void zklang::populateZhlComponentToLlzkStructConversionPatternsAndLegality(
    const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target
) {
  populateZhlComponentToLlzkStructConversionPatterns(tc, patterns);
  populateZhlComponentToLlzkStructConversionTarget(target);
}
