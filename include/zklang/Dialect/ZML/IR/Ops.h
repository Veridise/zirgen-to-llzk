#pragma once

#include "zklang/Dialect/ZML/IR/OpInterfaces.h" // IWYU pragma: keep
#include "zklang/Dialect/ZML/IR/Types.h"        // IWYU pragma: keep
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace zkc::Zmir {

/// Tag for ComponentOp build method
struct IsBuiltIn {};

} // namespace zkc::Zmir

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "zklang/Dialect/ZML/IR/Ops.h.inc"
