#pragma once

#include "zklang/Dialect/ZHL/Typing/FrameSlot.h"
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

namespace zhl {

class ComponentSlot : public FrameSlot {
public:
  /// Instantiates a slot for a component. Marks the input binding with the slot assigned to.
  ComponentSlot(TypeBinding &binding);
  ComponentSlot(TypeBinding &binding, mlir::StringRef name);

  static bool classof(const FrameSlot *);

  const TypeBinding &getBinding() const { return binding; }

private:
  TypeBinding binding;
};

} // namespace zhl
