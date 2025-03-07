#pragma once

#include <cassert>
#include <cstdint>
#include <deque>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string_view>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>

namespace llvm {
class raw_ostream;
}

namespace zhl {

class TypeBindings {
public:
  explicit TypeBindings(mlir::Location);

  const TypeBinding &Component();
  const TypeBinding &Component() const;
  const TypeBinding &Bottom() const;
  TypeBinding Const(uint64_t value, mlir::Location loc) const;
  TypeBinding UnkConst(mlir::Location loc) const;
  TypeBinding Array(TypeBinding type, uint64_t size, mlir::Location loc) const;
  TypeBinding Array(TypeBinding type, TypeBinding size, mlir::Location loc) const;
  TypeBinding UnkArray(TypeBinding type, mlir::Location loc) const;

  TypeBinding Const(uint64_t value) const;
  TypeBinding UnkConst() const;
  TypeBinding Array(TypeBinding type, uint64_t size) const;
  TypeBinding Array(TypeBinding type, TypeBinding size) const;
  TypeBinding UnkArray(TypeBinding type) const;

  [[nodiscard]] bool Exists(mlir::StringRef name) const;

  template <typename... Args>
  const TypeBinding &Create(mlir::StringRef name, mlir::Location loc, Args &&...args) {
    assert(bindings.find(name) == bindings.end() && "double binding write");
    bindings.try_emplace(name, TypeBinding(name, loc, std::forward<Args>(args)...));
    return bindings.at(name);
  }

  template <typename... Args> const TypeBinding &Create(mlir::StringRef name, Args &&...args) {
    return Create(name, unk, std::forward<Args>(args)...);
  }

  /// Creates a type binding and keeps track of its memory, but it is not registered in the
  /// named bindings table.
  template <typename... Args>
  const TypeBinding &CreateAnon(mlir::StringRef name, mlir::Location loc, Args &&...args) {
    return Manage(TypeBinding(name, loc, std::forward<Args>(args)...));
  }

  template <typename... Args> const TypeBinding &CreateAnon(mlir::StringRef name, Args &&...args) {
    return CreateAnon(name, unk, std::forward<Args>(args)...);
  }

  template <typename... Args>
  const TypeBinding &CreateBuiltin(mlir::StringRef name, mlir::Location loc, Args &&...args) {
    assert(bindings.find(name) == bindings.end() && "double binding write");
    bindings.try_emplace(name, TypeBinding(name, loc, std::forward<Args>(args)..., Frame(), true));
    return bindings.at(name);
  }

  template <typename... Args>
  const TypeBinding &CreateBuiltin(mlir::StringRef name, Args &&...args) {
    return CreateBuiltin(name, unk, std::forward<Args>(args)...);
  }

  [[nodiscard]] const TypeBinding &Get(mlir::StringRef name) const;
  [[nodiscard]] mlir::FailureOr<TypeBinding> MaybeGet(mlir::StringRef name) const;
  [[nodiscard]] const TypeBinding &Manage(const TypeBinding &);

private:
  mlir::Location unk;
  llvm::StringMap<TypeBinding> bindings;
  std::deque<TypeBinding> managedBindings;
  TypeBinding bottom;
};

} // namespace zhl

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const zhl::TypeBinding::Name &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const zhl::TypeBinding &b);
