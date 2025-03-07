#include <cassert>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>

#define DEBUG_TYPE "zhl-type-bindings"

using namespace zhl;
using namespace mlir;

namespace zhl {

struct TypeBinding::Name::Impl {
  Impl(mlir::StringRef nameRef) : name(nameRef) {}

  llvm::SmallString<10> name;
};

TypeBinding::Name::Name(const Name &) = default;
TypeBinding::Name &TypeBinding::Name::operator=(const Name &) = default;
TypeBinding::Name::Name(Name &&) = default;
TypeBinding::Name &TypeBinding::Name::operator=(Name &&) = default;

TypeBinding::Name::Name(mlir::StringRef name) : impl(std::make_shared<Impl>(name)) {}

TypeBinding::Name::~Name() = default;

TypeBinding::Name &TypeBinding::Name::operator=(mlir::StringRef newName) {
  impl->name = newName;
  return *this;
}

TypeBinding::Name::operator mlir::StringRef() const { return impl->name; }

StringRef TypeBinding::Name::ref() const { return impl->name; }

bool TypeBinding::Name::operator==(const TypeBinding::Name &other) const {
  return ref() == other.ref();
}

bool TypeBinding::Name::operator==(mlir::StringRef s) const { return ref() == s; }

} // namespace zhl
