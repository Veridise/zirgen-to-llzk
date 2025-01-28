#include "zklang/Dialect/ZML/IR/Dialect.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include "zklang/Dialect/ZML/Typing/Materialize.h"
#include <gtest/gtest.h>

//=----------------------------------------------------------=//
//  Materialization tests
//=----------------------------------------------------------=//

using namespace zhl;
using namespace mlir;
using namespace zkc;

class MaterializationTest : public testing::Test {
protected:
  MaterializationTest() : ctx{} /*, builder(&ctx)*/ /*, bindings(builder.getUnknownLoc())*/ {
    // ctx.loadDialect<Zmir::ZmirDialect>();
  }

  MLIRContext ctx;
  // OpBuilder builder;
  /*TypeBindings bindings;*/
};

TEST_F(MaterializationTest, componentBaseType) {
  /*Type output = Zmir::materializeTypeBinding(&ctx, bindings.Component());*/
  /*Type expected = Zmir::ComponentType::Component(&ctx);*/
  /*ASSERT_EQ(output, expected);*/
}
