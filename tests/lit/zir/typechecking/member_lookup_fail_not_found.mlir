// RUN: zklang-opt --inject-builtins --zhl-print-type-bindings -verify-diagnostics %s

// component A() {
//   x := Reg(1);
// }
// component Top() {
//   a := A();
//   a.y = 1;
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
    %0 = zhl.global "Reg"
    %1 = zhl.literal 1
    %2 = zhl.construct %0(%1)
    %3 = zhl.declare "x" {isPublic = false}
    zhl.define %3 = %2
    %4 = zhl.global "Component"
    %5 = zhl.construct %4()
    zhl.super %5
  }
  zhl.component @Top {
    %0 = zhl.global "A"
    %1 = zhl.construct %0()
    %2 = zhl.declare "a" {isPublic = false}
    zhl.define %2 = %1
    // expected-error@+2 {{member A.y was not found}}
    // expected-error@+1 {{could not deduce the type of op 'zhl.lookup'}}
    %3 = zhl.lookup %1, "y"
    %4 = zhl.literal 1
    zhl.constrain %3 = %4
    %5 = zhl.global "Component"
    %6 = zhl.construct %5()
    zhl.super %6
  }
}
