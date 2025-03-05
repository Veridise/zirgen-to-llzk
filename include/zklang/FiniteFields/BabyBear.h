#pragma once

#include <zklang/FiniteFields/Field.h>

namespace ff {

namespace babybear {

class Field : public FieldData {
public:
  Field() : FieldData(15 * (1 << 27) + 1, 4, 11) {}
};

} // namespace babybear

} // namespace ff
