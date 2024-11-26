#ifndef ZMIR_OP_INTERFACES
#define ZMIR_OP_INTERFACES

include "mlir/IR/Interfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Class for common structure for the interface.
// Implementors must be both Symbols and SymbolTables
class DefinesFunc<string ifaceName, string funcName>
    : OpInterface<ifaceName, [Symbol]> {
  let description =
      "Interface for accesing a function named "#funcName#" defined inside the "
      "implementor";
  let cppNamespace = "::zkc::Zmir";

  string capitalizedName;
  string ifaceMethodName;

  let methods = [InterfaceMethod<"Returns the actual name of the function",
                                 "mlir::StringRef", ifaceMethodName, (ins)>];

  let extraSharedClassDeclaration = [{
    mlir::SymbolRefAttr get}]#capitalizedName#[{Sym() {
      ::mlir::OpBuilder builder($_op->getContext());
      ::mlir::StringAttr parent = $_op->template getAttrOfType<::mlir::StringAttr>(::mlir::SymbolTable::getSymbolAttrName());

      return mlir::SymbolRefAttr::get(parent, {mlir::SymbolRefAttr::get(builder.getStringAttr($_op.}]#ifaceMethodName#[{()))});
    }

    mlir::func::FuncOp get}]#capitalizedName#[{Func() {
      mlir::OpBuilder builder($_op->getContext());
      return mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>($_op, builder.getStringAttr($_op.}]#ifaceMethodName#[{()));
    }
  }];
}

def DefinesBodyFunc : DefinesFunc<"DefinesBodyFunc", "body"> {
  let capitalizedName = "Body";
  let ifaceMethodName = "getBodyFuncName";
}

def DefinesConstrainFunc : DefinesFunc<"DefinesConstrainFunc", "constrain"> {
  let capitalizedName = "Constrain";
  let ifaceMethodName = "getConstrainFuncName";
}

def ComponentInterface
    : OpInterface<"ComponentInterface", [DefinesBodyFunc,
                                         DefinesConstrainFunc]> {
  let description =
      "Abstracts the public API of a component op from the implementations";

  let cppNamespace = "::zkc::Zmir";
  let methods =
      [InterfaceMethod<"Returns the type of the component", "::mlir::Type",
                       "getType", (ins)>,
       InterfaceMethod<"Returns the type of the super component",
                       "::mlir::Type", "getSuperType", (ins)>,
       InterfaceMethod<"Looks up the type of a field inside the component",
                       "::mlir::FailureOr<::mlir::Type>", "lookupFieldType",
                       (ins "::mlir::FlatSymbolRefAttr":$field)>,
       InterfaceMethod<"Returns wether the component is a builtin or not",
                       "bool", "getBuiltin", (ins)>,
       InterfaceMethod<
           "Returns the region where the component elements are defined",
           "::mlir::Region &", "getRegion", (ins)>,
       InterfaceMethod<
           "Returns whether the component has an unified body and constrains",
           "bool", "hasUnifiedBody", (ins)>];
}

#endif // ZKIR_OP_INTERFACES
