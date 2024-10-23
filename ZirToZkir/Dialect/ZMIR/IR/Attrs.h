#pragma once

#ifndef _ZMIR_ATTRS_H
#define _ZMIR_ATTRS_H

#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"

#include <mlir/IR/DialectImplementation.h>

// Include TableGen'd declarations
#define GET_ATTRDEF_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/Attrs.h.inc"

#endif
