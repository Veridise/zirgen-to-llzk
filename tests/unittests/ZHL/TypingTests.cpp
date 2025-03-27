#include "zklang/Dialect/ZML/BuiltIns/BuiltIns.h"
#include "zklang/Dialect/ZML/IR/Dialect.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include "zklang/Dialect/ZML/Typing/Materialize.h"
#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include <string_view>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

//=----------------------------------------------------------=//
//  Materialization tests
//=----------------------------------------------------------=//

using namespace zhl;
using namespace mlir;

namespace zhl {

/// Makes gtest about to print TypeBinding instances when reporting
/// test results
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
    ctx.loadDialect<zml::ZMLDialect>();
  }

  MLIRContext ctx;
  OpBuilder builder;
  TypeBindings bindings;
};

class MaterializationTestWithBuiltins : public MaterializationTest {
protected:
  MaterializationTestWithBuiltins() : MaterializationTest() {
    std::unordered_set<std::string_view> s;
    zml::addBuiltinBindings(bindings, s);
  }
};

TEST_F(MaterializationTest, componentBaseType) {
  Type output = zml::materializeTypeBinding(&ctx, bindings.Component());
  Type expected = zml::ComponentType::Component(&ctx);
  ASSERT_EQ(output, expected);
}

TEST_F(MaterializationTest, componentBaseConstructorType) {
  FunctionType output =
      zml::materializeTypeBindingConstructor(builder, bindings.Component(), bindings);
  FunctionType expected = builder.getFunctionType(TypeRange(), zml::ComponentType::Component(&ctx));
  ASSERT_EQ(output, expected);
}

//=----------------------------------------------------------=//
//  Specialization tests
//=----------------------------------------------------------=//

class SpecializationTest : public testing::Test {
protected:
  SpecializationTest() : ctx{}, bindings(unkLoc()) {
    std::unordered_set<std::string_view> s;
    zml::addBuiltinBindings(bindings, s);
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
      "Foo", Component, zhl::ParamsMap().declare("T", T), zhl::ParamsMap(), zhl::MembersMap()
  );
  TypeBinding ExpectedSpecializedFoo(
      "Foo", unkLoc(), Component, zhl::ParamsMap().declare("T", Val), zhl::ParamsMap(),
      zhl::MembersMap()
  );
  ExpectedSpecializedFoo.markAsSpecialized();
  auto SpecializedFoo = Foo.specialize(diag, {Val}, bindings);
  ASSERT_TRUE(succeeded(SpecializedFoo));
  ASSERT_EQ(*SpecializedFoo, ExpectedSpecializedFoo);
}

TEST_F(SpecializationTest, specializationPropagatesProperlyForArrays) {
  auto &Component = bindings.Component();
  auto &Type = bindings.Get("Type");
  auto T = TypeBinding::MakeGenericParam(Type, "T");
  auto Const10 = bindings.Const(10);
  auto &Val = bindings.Get("Val");
  auto &Arr = bindings.Get("Array");
  TypeBinding ExpectedSpecializedArray(
      "Array", unkLoc(), Component, zhl::ParamsMap().declare("T", Val).declare("N", Const10),
      zhl::ParamsMap(), zhl::MembersMap(), Frame(), true
  );
  ExpectedSpecializedArray.markAsSpecialized();
  ExpectedSpecializedArray.selfConstructs();
  auto SpecializedArr = Arr.specialize(diag, {Val, Const10}, bindings);
  ASSERT_TRUE(succeeded(SpecializedArr));
  ASSERT_EQ(*SpecializedArr, ExpectedSpecializedArray);
}

TEST_F(SpecializationTest, constValSpecializationPropagatesProperlyForArraysOfArraysCtorParam) {
  auto &Type = bindings.Get("Type");
  auto &Val = bindings.Get("Val");
  auto T = TypeBinding::MakeGenericParam(Type, "T");
  auto N = TypeBinding::MakeGenericParam(Val, "N");
  auto Const10 = bindings.Const(10);

  auto genericCtorParam = bindings.Array(bindings.Array(Val, N), N);

  TypeBinding GenericFoo(
      "Foo", unkLoc(), Val, zhl::ParamsMap().declare("N", N),
      zhl::ParamsMap().declare("m", genericCtorParam), zhl::MembersMap(), Frame(), true
  );

  auto ctorParam = bindings.Array(bindings.Array(Val, 10), 10);
  TypeBinding ExpectedType(
      "Foo", unkLoc(), Val, zhl::ParamsMap().declare("N", Const10),
      zhl::ParamsMap().declare("m", ctorParam), zhl::MembersMap(), Frame(), true
  );
  ExpectedType.markAsSpecialized();
  auto SpecializedFoo = GenericFoo.specialize(diag, {Const10}, bindings);
  ASSERT_TRUE(succeeded(SpecializedFoo));
  ASSERT_EQ(*SpecializedFoo, ExpectedType);
}

/// Tests that the specialization specializes the constructor params of a type
TEST_F(SpecializationTest, specializationPropagatesProperlyToConstructor) {
  auto &Component = bindings.Component();
  auto &Type = bindings.Get("Type");
  auto T = TypeBinding::MakeGenericParam(Type, "T");
  auto &Val = bindings.Get("Val");
  auto &Foo = bindings.Create(
      "Foo", Component, zhl::ParamsMap().declare("T", T), zhl::ParamsMap().declare("t", T),
      zhl::MembersMap()
  );
  TypeBinding ExpectedSpecializedFoo(
      "Foo", unkLoc(), Component, zhl::ParamsMap().declare("T", Val),
      zhl::ParamsMap().declare("t", Val), zhl::MembersMap()
  );
  ExpectedSpecializedFoo.markAsSpecialized();
  auto SpecializedFoo = Foo.specialize(diag, {Val}, bindings);
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
      "Foo", Component, zhl::ParamsMap().declare("T", T), zhl::ParamsMap(), zhl::MembersMap()
  );
  auto Foo_F = Foo.specialize(diag, {F}, bindings); // The specialized version of Foo within Bar.
  ASSERT_TRUE(succeeded(Foo_F));
  auto &Bar = bindings.Create(
      "Bar", *Foo_F, zhl::ParamsMap().declare("F", F), zhl::ParamsMap(), zhl::MembersMap()
  );
  // The supertype should be like this after specializing Bar
  TypeBinding ExpectedSpecializedFoo(
      "Foo", unkLoc(), Component, zhl::ParamsMap().declare("T", Val), zhl::ParamsMap(),
      zhl::MembersMap()
  );
  ExpectedSpecializedFoo.markAsSpecialized();

  auto SpecializedBar = Bar.specialize(diag, {Val}, bindings);
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
      "Foo", Component, zhl::ParamsMap().declare("T", T), zhl::ParamsMap(),
      zhl::MembersMap({{"t", T}})
  );
  TypeBinding ExpectedSpecializedFoo(
      "Foo", unkLoc(), Component, zhl::ParamsMap().declare("T", Val), zhl::ParamsMap(),
      zhl::MembersMap({{"t", Val}})
  );
  ExpectedSpecializedFoo.markAsSpecialized();
  auto SpecializedFoo = Foo.specialize(diag, {Val}, bindings);
  ASSERT_TRUE(succeeded(SpecializedFoo));
  ASSERT_EQ(*SpecializedFoo, ExpectedSpecializedFoo);
}

TEST(TypeBindingTest, NameChangesShouldPropagateToSharedCopies) {
  TypeBinding::Name orig("Test");
  TypeBinding::Name copy = orig;

  ASSERT_EQ(orig.ref(), "Test");
  ASSERT_EQ(copy.ref(), "Test");

  orig = "Changed";
  ASSERT_EQ(orig.ref(), "Changed");
  ASSERT_EQ(copy.ref(), "Changed");
}
