// RUN: zklang-opt  %s -verify-diagnostics --inject-builtins --lower-zhl

// component Bar<V: Val>(arr: Array<Val, V>) {
//   arr[0]
// }
// component Foo(v: Val) {
//   bar := Bar<v>([1,2,3]);
// }

module {
  zhl.component @Reg {
    %0 = zhl.global "Val"
    %1 = zhl.parameter "v"(0) : %0
    %2 = zhl.global "NondetReg"
    %3 = zhl.construct %2(%1)
    %4 = zhl.declare "reg" {isPublic = false}
    zhl.define %4 = %3
    zhl.constrain %1 = %3
    zhl.super %3
  }
  zhl.component @Div attributes {function} {
    %0 = zhl.global "Val"
    %1 = zhl.parameter "lhs"(0) : %0
    %2 = zhl.global "Val"
    %3 = zhl.parameter "rhs"(1) : %2
    %4 = zhl.global "Inv"
    %5 = zhl.construct %4(%3)
    %6 = zhl.declare "reciprocal" {isPublic = false}
    zhl.define %6 = %5
    %7 = zhl.global "Mul"
    %8 = zhl.construct %7(%5, %3)
    %9 = zhl.literal 1
    zhl.constrain %8 = %9
    %10 = zhl.global "Mul"
    %11 = zhl.construct %10(%5, %1)
    zhl.super %11
  }
  zhl.component @Log {
    %0 = zhl.global "String"
    %1 = zhl.parameter "message"(0) : %0
    %2 = zhl.global "Val"
    %3 = zhl.parameter "vals"(1) : %2 {variadic}
    %4 = zhl.global "Component"
    %5 = zhl.extern "Log"(%1, %3) : %4
    zhl.super %5
  }
  zhl.component @Bar attributes {generic} {
    %0 = zhl.global "Val"
    %1 = zhl.generic "V"(0) : %0
    %2 = zhl.global "Array"
    %3 = zhl.global "Val"
    %4 = zhl.specialize %2<%3, %1>
    %5 = zhl.parameter "arr"(0) : %4
    %6 = zhl.literal 0
    %7 = zhl.subscript %5[%6]
    zhl.super %7
  }
  zhl.component @Foo {
    %0 = zhl.global "Val"
    %1 = zhl.parameter "v"(0) : %0
    %2 = zhl.global "Bar"
      // expected-error@+2 {{was expecting a type, literal value, constant expression or generic parameter, but got a value}}
      // expected-error@+1 {{could not deduce the type of op 'zhl.specialize'}}
    %3 = zhl.specialize %2<%1>
    %4 = zhl.literal 1
    %5 = zhl.literal 2
    %6 = zhl.literal 3
    %7 = zhl.array[%4, %5, %6]
    %8 = zhl.construct %3(%7)
    %9 = zhl.declare "bar" {isPublic = false}
    zhl.define %9 = %8
    %10 = zhl.global "Component"
    %11 = zhl.construct %10()
    zhl.super %11
  }
}
