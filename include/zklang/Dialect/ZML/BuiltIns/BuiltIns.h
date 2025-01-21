#pragma once

#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include <mlir/IR/Builders.h>
#include <string_view>
#include <unordered_set>

namespace zkc::Zmir {

void addBuiltinBindings(zhl::TypeBindings &, const std::unordered_set<std::string_view> &);
// Add builtin components using the given builder
void addBuiltins(mlir::OpBuilder &, const std::unordered_set<std::string_view> &);
mlir::Operation *getBuiltInOp(mlir::StringRef);

static const std::unordered_set<std::string> BuiltInComponentNames = {
    "BitAnd", "Add", "Sub", "Mul", "Inv", "Isz", "Neg", "Val", "String", "Array"
};

static const char BitAndStr[] = "BitAnd";
static const char AddStr[] = "Add";
static const char SubStr[] = "Sub";
static const char MulStr[] = "Mul";
static const char InvStr[] = "Inv";
static const char IszStr[] = "Isz";
static const char NegStr[] = "Neg";
static const char ValStr[] = "Val";
static const char StrStr[] = "String";
static const char ComponentStr[] = "Component";
static const char ArrayStr[] = "Array";

} // namespace zkc::Zmir
