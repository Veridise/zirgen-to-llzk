#pragma once

#include "ZirToZkir/Dialect/ZMIR/IR/OpInterfaces.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>

namespace zkc::Zmir {

bool isValidZmirType(mlir::Type);

}

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.inc.h"

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/Types.inc.h"
