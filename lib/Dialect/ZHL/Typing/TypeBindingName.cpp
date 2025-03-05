#include <cassert>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

#define DEBUG_TYPE "zhl-type-bindings"

using namespace zhl;
using namespace mlir;

namespace zhl {

struct TypeBindingName::Impl {
  Impl(mlir::StringRef nameRef) : name(nameRef) {}

  std::string name;
};

TypeBindingName::TypeBindingName(const TypeBindingName &) = default;
TypeBindingName &TypeBindingName::operator=(const TypeBindingName &) = default;
TypeBindingName::TypeBindingName(TypeBindingName &&) = default;
TypeBindingName &TypeBindingName::operator=(TypeBindingName &&) = default;

TypeBindingName::TypeBindingName(mlir::StringRef name) : impl(std::make_shared<Impl>(name)) {}

TypeBindingName::~TypeBindingName() = default;

TypeBindingName &TypeBindingName::operator=(mlir::StringRef newName) {
  impl->name = newName;
  return *this;
}

TypeBindingName::operator mlir::StringRef() const { return impl->name; }

StringRef TypeBindingName::ref() const { return impl->name; }

bool TypeBindingName::operator==(const TypeBindingName &other) const {
  return ref() == other.ref();
}

bool TypeBindingName::operator==(mlir::StringRef s) const { return ref() == s; }

} // namespace zhl
