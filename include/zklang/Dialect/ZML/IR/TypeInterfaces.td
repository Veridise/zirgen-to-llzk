#ifndef ZMIR_TYPE_INTERFACES
#define ZMIR_TYPE_INTERFACES

include "mlir/IR/Interfaces.td"

def Array : TypeInterface<"ArrayType"> {
  let description = "common interface for array types";

  let cppNamespace = "::zkc::Zmir";
  let methods = [InterfaceMethod<"Returns the type of the array",
                                 "::mlir::Type", "getInnerType", (ins)>,
                 InterfaceMethod<"Returns the size of the array",
                                 "mlir::Attribute", "getSize", (ins)>,
                 InterfaceMethod<"Returns the size of the array", "int64_t",
                                 "getSizeInt", (ins)>];
}

#endif // ZKIR_OP_INTERFACES
