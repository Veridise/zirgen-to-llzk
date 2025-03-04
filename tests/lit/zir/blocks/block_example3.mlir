// RUN: zklang-opt  %s -verify-diagnostics --inject-builtins --lower-zhl

// A name that is not available in the component's local scope is assumed to be a type name.

// Original ZIR code:
// component A() {
//   a := 1;
//   b := {
//    x := a;
//    x
//   };
//   c := x;
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
  zhl.component @A {
    %0 = zhl.literal 1
    %1 = zhl.declare "a" {isPublic = false}
    zhl.define %1 = %0
    %2 = zhl.block {
      %8 = zhl.declare "x" {isPublic = false}
      zhl.define %8 = %0
      zhl.super %0
    }
    %3 = zhl.declare "b" {isPublic = false}
    zhl.define %3 = %2
      // expected-error@+2 {{could not deduce the type of op 'zhl.global'}}
      // expected-error@+1 {{type 'x' was not found}}
    %4 = zhl.global "x"
    %5 = zhl.declare "c" {isPublic = false}
    zhl.define %5 = %4
    %6 = zhl.global "Component"
    %7 = zhl.construct %6()
    zhl.super %7
  }
}
