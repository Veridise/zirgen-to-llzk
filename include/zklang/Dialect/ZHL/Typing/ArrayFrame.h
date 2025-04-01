#pragma once

#include <mlir/IR/Value.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>

namespace zhl {

/// Represents a slot with an inner frame where every
/// slot in the inner frame is repeat N times by using an Array.
/// Is tied to an induction variable that represents the current iteration
/// of the loop this frame is tied to. It must be set before
/// child slots can be retrieved.
class ArrayFrame : public ComponentSlot {
public:
  ArrayFrame(const TypeBindings &);
  Frame &getFrame();

  static bool classof(const FrameSlot *);

  void setInductionVar(mlir::Value);
  mlir::Value getInductionVar() const;

  void setSize(const TypeBinding &Size) { size = Size; }
  const TypeBinding &getSize() const { return size; }

  void print(llvm::raw_ostream &) const override;

protected:
  mlir::StringRef defaultNameForTemporaries() const override;

private:
  mlir::Value iv;
  Frame innerFrame;
  TypeBinding size;
};

} // namespace zhl
