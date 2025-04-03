//===- Field.h - Finite fields data -----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a struct that holds information about a finite field.
//
//===----------------------------------------------------------------------===//

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
