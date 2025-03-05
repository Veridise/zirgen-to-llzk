#pragma once

#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>

namespace zhl {

/// Represents a Frame inside another
class InnerFrame : public ComponentSlot {
public:
  InnerFrame(const TypeBindings &);
  Frame &getFrame();

  static bool classof(const FrameSlot *);

  void print(llvm::raw_ostream &) const override;

private:
  Frame innerFrame;
};

} // namespace zhl
