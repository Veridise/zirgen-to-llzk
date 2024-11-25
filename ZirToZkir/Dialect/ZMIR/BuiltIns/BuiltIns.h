#pragma once

#include <mlir/IR/Builders.h>
#include <unordered_set>

namespace zkc::Zmir {

// Add builtin components using the given builder
void addBuiltins(mlir::OpBuilder &, std::function<bool(mlir::StringRef)>);
mlir::Operation *getBuiltInOp(mlir::StringRef);

static const std::unordered_set<std::string> BuiltInComponentNames = {
    "BitAnd", "Add", "Sub", "Mul", "Inv", "Isz", "Neg"};

static const char BitAndStr[] = "BitAnd";
static const char AddStr[] = "Add";
static const char SubStr[] = "Sub";
static const char MulStr[] = "Mul";
static const char InvStr[] = "Inv";
static const char IszStr[] = "Isz";
static const char NegStr[] = "Neg";

} // namespace zkc::Zmir
