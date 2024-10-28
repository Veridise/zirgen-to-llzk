#pragma once

#include "ZirToZkir/Dialect/ZMIR/IR/Attrs.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>

#include <llvm/ADT/TypeSwitch.h>

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h.inc"

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h.inc"
