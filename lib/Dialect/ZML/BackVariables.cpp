#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Utils/BackVariables.h>

using namespace zml;
using namespace mlir;

Type ZML_BVTypeProvider::getFieldType(FlatSymbolRefAttr name, Operation *op) const {
  assert(op && "Need to provide an operation");
  if (auto fieldOp = mlir::dyn_cast<FieldDefOp>(op)) {
    assert(fieldOp.getName() == name.getValue() && "op is a field def with a different name");
    return fieldOp.getType();
  }

  llvm_unreachable("Reached a field search case that is not handled");
  return nullptr;
}
