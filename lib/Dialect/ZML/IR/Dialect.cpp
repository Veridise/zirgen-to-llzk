//===- Dialect.cpp - ZML Dialect --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Felt/IR/Types.h>
#include <llzk/Dialect/Polymorphic/IR/Types.h>
#include <llzk/Dialect/String/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
#include <zklang/Dialect/ZML/IR/Attrs.h> // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>
#include <zklang/Dialect/ZML/IR/Ops.h> // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>

// TableGen'd implementation files
#include <zklang/Dialect/ZML/IR/Dialect.cpp.inc>

// Need a complete declaration of storage classes
#define GET_TYPEDEF_CLASSES
#include <zklang/Dialect/ZML/IR/Types.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <zklang/Dialect/ZML/IR/Attrs.cpp.inc>

using namespace zml;
using namespace mlir;

auto ZMLDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "zklang/Dialect/ZML/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "zklang/Dialect/ZML/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "zklang/Dialect/ZML/IR/Attrs.cpp.inc"
  >();
  // clang-format on

  declarePromisedInterface<llzk::component::StructDefOp, DefinesBodyFunc>();
  declarePromisedInterface<llzk::component::StructDefOp, DefinesConstrainFunc>();
  declarePromisedInterface<llzk::component::StructDefOp, ComponentInterface>();
  declarePromisedInterface<llzk::felt::FeltType, ComponentLike>();
  declarePromisedInterface<llzk::string::StringType, ComponentLike>();
  declarePromisedInterface<llzk::array::ArrayType, ComponentLike>();
  declarePromisedInterface<llzk::polymorphic::TypeVarType, ComponentLike>();
}

//===----------------------------------------------------------------------===//
// FixedTypeBindingAttr
//===----------------------------------------------------------------------===//

FixedTypeBindingAttr FixedTypeBindingAttr::get(
    MLIRContext *ctx, const zhl::TypeBinding &binding, const zhl::TypeBindings &bindings
) {
  Builder builder(ctx);

  auto Type = materializeTypeBinding(ctx, binding);
  auto Name = builder.getStringAttr(binding.getName());
  auto GenericParams =
      llvm::map_to_vector(binding.getGenericParamsMapping().zipped(), [ctx, &bindings](auto param) {
    Builder builder(ctx);
    auto [ParamBinding, ParamName, IsInjected] = param;
    return GenericParamAttr::get(
        ctx, builder.getStringAttr(ParamName),
        FixedTypeBindingAttr::get(ctx, ParamBinding, bindings), IsInjected
    );
  });
  auto Members = llvm::map_to_vector(binding.getMembers(), [ctx, &bindings](auto &member) {
    Builder builder(ctx);

    return ComponentMemberAttr::get(
        ctx, builder.getStringAttr(member.getKey()),
        FixedTypeBindingAttr::get(ctx, *member.getValue(), bindings),
        true // TODO: Add detection of public or private according to the new model
    );
  });
  auto CtorType = materializeTypeBindingConstructor(builder, binding, bindings);
  auto ParamLocs =
      llvm::map_to_vector(binding.getConstructorParamLocations(), [](Location loc) -> LocationAttr {
    return loc;
  });
  FixedTypeBindingAttr SuperType = nullptr;
  if (binding.hasSuperType()) {
    SuperType = FixedTypeBindingAttr::get(ctx, binding.getSuperType(), bindings);
  }
  Attribute ConstExpr = nullptr;
  if (auto expr = binding.getConstExpr()) {
    ConstExpr = expr->convertIntoAttribute(builder);
  }

  return Base::get(
      ctx, Type, Name, GenericParams, Members, CtorType, ParamLocs, SuperType, ConstExpr,
      binding.isBuiltin(), binding.isExtern()
  );
}

SmallVector<StringRef> FixedTypeBindingAttr::getGenericParamNames() const {
  return llvm::map_to_vector(getGenericParams(), [](auto param) {
    return param.getName().getValue();
  });
}
