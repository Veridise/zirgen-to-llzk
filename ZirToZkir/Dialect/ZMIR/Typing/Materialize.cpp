#include "Materialize.h"
#include "ZirToZkir/Dialect/ZHL/Typing/TypeBindings.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include <algorithm>
#include <iterator>
#include <mlir/IR/MLIRContext.h>

using namespace mlir;
using namespace zhl;

namespace zkc::Zmir {

Type materializeTypeBinding(MLIRContext *context, const TypeBinding &binding) {
  if (binding.isBottom()) {
    return nullptr;
  }

  auto materializeBaseType = [&]() -> mlir::Type {
    if (binding.isGeneric()) {
      if (!binding.isSpecialized()) {
        return nullptr;
      }

      std::vector<Attribute> params; // TODO: Fill this up
      return ComponentType::get(context, binding.getName(), params);
    } else if (binding.isConst()) {
      return ComponentType::get(context, "Val");
    } else {
      return ComponentType::get(context, binding.getName());
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
