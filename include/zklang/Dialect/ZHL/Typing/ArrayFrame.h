#pragma once

#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
namespace zhl {

/// Represents a slot with an inner frame where every
/// slot in the inner frame is repeat N times by using an Array.
class ArrayFrame : public FrameSlot {
public:
  ArrayFrame();
  Frame &getFrame();

  static bool classof(const FrameSlot *);

private:
  Frame innerFrame;
};

} // namespace zhl
