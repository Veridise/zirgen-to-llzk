#pragma once

#include <cstdint>

namespace ff {

struct FieldData {

  virtual ~FieldData() = default;

  uint64_t degree, prime, beta;

protected:
  FieldData(uint64_t prime_, uint64_t degree_, uint64_t beta_)
      : degree(degree_), prime(prime_), beta(beta_) {}
};

} // namespace ff
