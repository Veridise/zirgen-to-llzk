#include "zklang/Dialect/ZML/BuiltIns/BuiltIns.h"
#include "zklang/Dialect/ZML/IR/Dialect.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include "zklang/Dialect/ZML/Typing/Materialize.h"
#include <gtest/gtest.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

//=----------------------------------------------------------=//
//  Materialization tests
//=----------------------------------------------------------=//

using namespace zhl;
using namespace mlir;
using namespace zkc;

namespace zhl {

void PrintTo(const TypeBinding &binding, std::ostream *os) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  binding.print(ss, true);
  *os << s;
}

} // namespace zhl

class MaterializationTest : public testing::Test {
protected:
  MaterializationTest() : ctx{}, builder(&ctx), bindings(builder.getUnknownLoc()) {
    ctx.loadDialect<Zmir::ZmirDialect>();
  }

  MLIRContext ctx;
  OpBuilder builder;
  TypeBindings bindings;
};

class MaterializationTestWithBuiltins : public MaterializationTest {
protected:
  MaterializationTestWithBuiltins() : MaterializationTest() {
    std::unordered_set<std::string_view> s;
    Zmir::addBuiltinBindings(bindings, s);
  }
};

TEST_F(MaterializationTest, componentBaseType) {
  Type output = Zmir::materializeTypeBinding(&ctx, bindings.Component());
  Type expected = Zmir::ComponentType::Component(&ctx);
  ASSERT_EQ(output, expected);
}

TEST_F(MaterializationTest, componentBaseConstructorType) {
  FunctionType output = Zmir::materializeTypeBindingConstructor(builder, bindings.Component());
  FunctionType expected =
      builder.getFunctionType(TypeRange(), Zmir::ComponentType::Component(&ctx));
  ASSERT_EQ(output, expected);
}

/*TEST_F(MaterializationTestWithBuiltins, genericCompWithArrayArgument) {*/
/*  auto &Val = bindings.Get("Val");*/
/*  auto Size = TypeBinding::MakeGenericParam(Val, "SIZE");*/
/*  auto &Array = bindings.Get("Array");*/
/*  auto ValArray = Array; // Copy to create the specialization*/
/*  ValArray.replaceGenericParamByName("T", Val);*/
/*  ValArray.replaceGenericParamByName("N", Size);*/
/*  // Foo<Size>(Array<Val(Val),Size>(Array<T,N>),Val(Val))*/
/*  auto &Foo = bindings.Create(*/
/*      "Foo", bindings.Component(), zhl::ParamsMap({{{"SIZE", 0}, Val}}),*/
/*      zhl::ParamsMap({{{"0", 0}, ValArray}, {{"1", 1}, Val}}), zhl::MembersMap()*/
/*  );*/
/*  FunctionType output = Zmir::materializeTypeBindingConstructor(builder, Foo);*/
/*  Type ValZmlType = Zmir::ComponentType::Val(&ctx);*/
/*  Attribute ValZmlTypeAttr = mlir::TypeAttr::get(ValZmlType);*/
/*  Attribute SIZE_var = mlir::SymbolRefAttr::get(mlir::StringAttr::get(&ctx, "SIZE"));*/
/*  Type ValArrayZmlType = Zmir::ComponentType::get(*/
/*      &ctx, "Array", Zmir::ComponentType::Component(&ctx), {ValZmlTypeAttr, SIZE_var}, true*/
/*  );*/
/*  Type FooZmlType = Zmir::ComponentType::get(*/
/*      &ctx, "Foo", Zmir::ComponentType::Component(&ctx), {SIZE_var}, false*/
/*  );*/
/*  FunctionType expected = builder.getFunctionType({ValArrayZmlType, ValZmlType}, FooZmlType);*/
/*  ASSERT_EQ(output, expected);*/
/*}*/

//=----------------------------------------------------------=//
//  Specialization tests
//=----------------------------------------------------------=//

class SpecializationTest : public testing::Test {
protected:
  SpecializationTest() : ctx{}, bindings(unkLoc()) {
    std::unordered_set<std::string_view> s;
    Zmir::addBuiltinBindings(bindings, s);
  }

  Location unkLoc() { return OpBuilder(&ctx).getUnknownLoc(); }

  MLIRContext ctx;
  TypeBindings bindings;
};
InFlightDiagnostic diag() { return InFlightDiagnostic(); }

/// Tests that the specialization replaces the generic params of a type
TEST_F(SpecializationTest, specializationPropagatesProperly) {
  auto &Component = bindings.Component();
  auto &Type = bindings.Get("Type");
  auto T = TypeBinding::MakeGenericParam(Type, "T");
  auto &Val = bindings.Get("Val");
  auto &Foo = bindings.Create(
      "Foo", Component, zhl::ParamsMap({{{"T", 0}, T}}), zhl::ParamsMap(), zhl::MembersMap()
  );
  TypeBinding ExpectedSpecializedFoo(
      "Foo", unkLoc(), Component, zhl::ParamsMap({{{"T", 0}, Val}}), zhl::ParamsMap(),
      zhl::MembersMap()
  );
  ExpectedSpecializedFoo.markAsSpecialized();
  auto SpecializedFoo = Foo.specialize(diag, {Val});
  ASSERT_TRUE(succeeded(SpecializedFoo));
  ASSERT_EQ(*SpecializedFoo, ExpectedSpecializedFoo);
}

/// Tests that the specialization specializes the constructor params of a type
TEST_F(SpecializationTest, specializationPropagatesProperlyToConstructor) {
  auto &Component = bindings.Component();
  auto &Type = bindings.Get("Type");
  auto T = TypeBinding::MakeGenericParam(Type, "T");
  auto &Val = bindings.Get("Val");
  auto &Foo = bindings.Create(
      "Foo", Component, zhl::ParamsMap({{{"T", 0}, T}}), zhl::ParamsMap({{{"t", 0}, T}}),
      zhl::MembersMap()
  );
  TypeBinding ExpectedSpecializedFoo(
      "Foo", unkLoc(), Component, zhl::ParamsMap({{{"T", 0}, Val}}),
      zhl::ParamsMap({{{"t", 0}, Val}}), zhl::MembersMap()
  );
  ExpectedSpecializedFoo.markAsSpecialized();
  auto SpecializedFoo = Foo.specialize(diag, {Val});
  ASSERT_TRUE(succeeded(SpecializedFoo));
  ASSERT_EQ(*SpecializedFoo, ExpectedSpecializedFoo);
}

/// Tests that the specialization also specializes the super type
TEST_F(SpecializationTest, specializationPropagatesProperlyToSuperType) {
  auto &Component = bindings.Component();
  auto &Type = bindings.Get("Type");
  auto T = TypeBinding::MakeGenericParam(Type, "T");
  auto F = TypeBinding::MakeGenericParam(Type, "F");
  auto &Val = bindings.Get("Val");
  auto &Foo = bindings.Create(
      "Foo", Component, zhl::ParamsMap({{{"T", 0}, T}}), zhl::ParamsMap(), zhl::MembersMap()
  );
  auto Foo_F = Foo.specialize(diag, {F}); // The specialized version of Foo within Bar.
  ASSERT_TRUE(succeeded(Foo_F));
  auto &Bar = bindings.Create(
      "Bar", *Foo_F, zhl::ParamsMap({{{"F", 0}, F}}), zhl::ParamsMap(), zhl::MembersMap()
  );
  // The supertype should be like this after specializing Bar
  TypeBinding ExpectedSpecializedFoo(
      "Foo", unkLoc(), Component, zhl::ParamsMap({{{"T", 0}, Val}}), zhl::ParamsMap(),
      zhl::MembersMap()
  );
  ExpectedSpecializedFoo.markAsSpecialized();

  auto SpecializedBar = Bar.specialize(diag, {Val});
  ASSERT_TRUE(succeeded(SpecializedBar));
  ASSERT_TRUE(SpecializedBar->hasSuperType());
  auto &super = SpecializedBar->getSuperType();
  ASSERT_EQ(super, ExpectedSpecializedFoo);
}

/// Tests that the specialization specializes the members of a type
TEST_F(SpecializationTest, specializationPropagatesProperlyToMembers) {
  auto &Component = bindings.Component();
  auto &Type = bindings.Get("Type");
  auto T = TypeBinding::MakeGenericParam(Type, "T");
  auto &Val = bindings.Get("Val");
  auto &Foo = bindings.Create(
      "Foo", Component, zhl::ParamsMap({{{"T", 0}, T}}), zhl::ParamsMap(),
      zhl::MembersMap({{"t", T}})
  );
  TypeBinding ExpectedSpecializedFoo(
      "Foo", unkLoc(), Component, zhl::ParamsMap({{{"T", 0}, Val}}), zhl::ParamsMap(),
      zhl::MembersMap({{"t", Val}})
  );
  ExpectedSpecializedFoo.markAsSpecialized();
  auto SpecializedFoo = Foo.specialize(diag, {Val});
  ASSERT_TRUE(succeeded(SpecializedFoo));
  ASSERT_EQ(*SpecializedFoo, ExpectedSpecializedFoo);
}
