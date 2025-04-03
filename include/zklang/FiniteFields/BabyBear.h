//===- BabyBear.h - BabyBear finite field -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a definition of the BabyBear finite field.
//
//===----------------------------------------------------------------------===//

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
