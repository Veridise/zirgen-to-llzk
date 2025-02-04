#pragma once

#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>

namespace zhl {

/// Represents a Frame inside another
class InnerFrame : public FrameSlot {
public:
  InnerFrame();
  Frame &getFrame();

  static bool classof(const FrameSlot *);

private:
  Frame innerFrame;
};

} // namespace zhl
