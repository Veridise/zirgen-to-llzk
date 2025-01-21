#pragma once

#include "zklang/Dialect/ZML/IR/OpInterfaces.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>

namespace zkc::Zmir {

bool isValidZmirType(mlir::Type);

}

#include "zklang/Dialect/ZML/IR/TypeInterfaces.h.inc"

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include "zklang/Dialect/ZML/IR/Ops.h.inc"

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "zklang/Dialect/ZML/IR/Types.h.inc"
