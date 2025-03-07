#pragma once

#include <utility>

namespace zklang {

/// A factory class that always returns null
template <typename Typ> class NullptrFactory {
public:
  static Typ *init() { return nullptr; }
};

/// A owning smart pointer that when copied creates a new pointer in the copy using the copy
/// constructor of the pointee type. Default initializes to the result of the factory method and by
/// default the factory returns nullptr.
template <typename Typ, typename Factory = NullptrFactory<Typ>> class CopyablePointer {
public:
  using self = CopyablePointer<Typ, Factory>;
  using type = Typ;
  using pointer = Typ *;
  using ref = Typ &;
  using cref = const Typ &;

  /// Default initialize the pointer to the result of the Factory.
  CopyablePointer() : ptr(Factory::init()) {}

  /// Initializes the pointer with the given pointer and adopts ownership of it.
  CopyablePointer(pointer Ptr) : ptr(Ptr) {}
  CopyablePointer &operator=(pointer other) {
    safeAssign(other);
    return *this;
  }

  CopyablePointer(const self &other) : ptr(safeCopy(other.ptr)) {}
  CopyablePointer &operator=(const self &other) {
    safeCopyAndDelete(other.ptr);
    return *this;
  }

  CopyablePointer(self &&other) : ptr(other.ptr) { other.ptr = Factory::init(); }
  CopyablePointer &operator=(self &&other) {
    if (this != &other) {
      safeCopyAndMove(other.ptr);
    }
    return *this;
  }

  ~CopyablePointer() { delete ptr; }

  pointer get() { return ptr; }
  const pointer get() const { return ptr; }

  void set(pointer Ptr) { safeAssign(Ptr); }

  pointer operator->() { return ptr; }
  const pointer operator->() const { return ptr; }

  ref operator*() {
    assert(ptr != nullptr);
    return *ptr;
  }

  cref operator*() const {
    assert(ptr != nullptr);
    return *ptr;
  }

  operator bool() const { return ptr != nullptr; }

  /// Referential equality
  bool operator==(const self &other) const { return ptr == other.ptr; }

#if 0
  /// Structural equality
  template <
      typename = std::enable_if_t<
          std::is_same_v<decltype(std::declval<cref>().operator==(std::declval<cref>())), bool>>>
  bool equalTo(const self &other) const {
    if (ptr == nullptr && other.ptr == nullptr) {
      return true;
    }
    if (ptr == nullptr || other.ptr == nullptr) {
      return false;
    }
    return *ptr == *other.ptr;
  }
#endif

  self clone() const { return self(*this); }

private:
  static pointer safeCopy(const pointer other) {
    return other ? new type(*other) : Factory::init();
  }

  void safeCopyAndDelete(const pointer other) {
    if (ptr == other) {
      return;
    }
    delete ptr;
    ptr = safeCopy(other);
  }

  void safeCopyAndMove(pointer &other) {
    if (safeAssign(other)) {
      other = Factory::init();
    }
  }

  bool safeAssign(pointer other) {
    if (ptr == other) {
      return false;
    }
    delete ptr;
    ptr = other;
    return true;
  }

  pointer ptr;
};

template <typename Typ, typename... Args, typename Factory = NullptrFactory<Typ>>
CopyablePointer<Typ, Factory> makeCopyablePointer(Args &&...args) {
  return CopyablePointer<Typ, Factory>(new Typ(std::forward<Args>(args)...));
}

} // namespace zklang
