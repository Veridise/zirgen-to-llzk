#include "Materialize.h"
#include "ZirToZkir/Dialect/ZHL/Typing/TypeBindings.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include <algorithm>
#include <iterator>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>

using namespace mlir;
using namespace zhl;

namespace zkc::Zmir {

Type materializeTypeBinding(MLIRContext *context, const TypeBinding &binding) {
  if (binding.isBottom()) {
    return nullptr;
  }

  auto materializeBaseType = [&]() -> mlir::Type {
    if (!binding.hasSuperType()) {
      return ComponentType::Component(context);
    }
    auto superType = materializeTypeBinding(context, binding.getSuperType());
    if (!mlir::isa<ComponentType>(superType)) {
      return nullptr;
    }
    auto superTypeComp = mlir::cast<ComponentType>(superType);
    if (binding.isGeneric()) {
      std::vector<Attribute> params;
      if (binding.isSpecialized()) {
        // Put the types associated with the specialization
        auto paramBindings = binding.getGenericParams();
        std::transform(
            paramBindings.begin(), paramBindings.end(), std::back_inserter(params),
            [&](const auto &b) -> Attribute {
          if (b.isConst()) {
            return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), b.getConst());
          } else {
            return mlir::TypeAttr::get(materializeTypeBinding(context, b));
          }
        }
        );
      } else {
        // Put the names of the parameters
        auto names = binding.getGenericParamNames();
        std::transform(
            names.begin(), names.end(), std::back_inserter(params),
            [&](const auto &name) { return SymbolRefAttr::get(StringAttr::get(context, name)); }
        );
      }

      return ComponentType::get(context, binding.getName(), superTypeComp, params);
    } else if (binding.isConst()) {
      return ComponentType::Val(context);
    } else {
      return ComponentType::get(context, binding.getName(), superTypeComp);
    }
  };

  return binding.isVariadic() ? VarArgsType::get(context, materializeBaseType())
                              : materializeBaseType();
}

FunctionType materializeTypeBindingConstructor(OpBuilder &builder, const TypeBinding &binding) {
  // Create the type of the binding and of each argument
  // then return a function type using the generated types.
  // If any of the given types is a null just return nullptr for the whole thing.
  std::vector<Type> args;
  auto retType = materializeTypeBinding(builder.getContext(), binding);
  if (!retType) {
    return nullptr;
  }

  auto params = binding.getConstructorParams();
  std::transform(params.begin(), params.end(), std::back_inserter(args), [&](auto &argBinding) {
    return materializeTypeBinding(builder.getContext(), argBinding);
  });
  if (std::any_of(args.begin(), args.end(), [](Type t) { return !t; })) {
    return nullptr;
  }

  return builder.getFunctionType(args, retType);
}

} // namespace zkc::Zmir
