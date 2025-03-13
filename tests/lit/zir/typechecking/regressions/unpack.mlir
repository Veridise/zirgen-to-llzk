// RUN: zklang-opt --zhl-print-type-bindings --verify-diagnostics %s

module {
  zhl.component @Reg {
    %0 = zhl.global "Val" // expected-remark{{type: Val}}
    %1 = zhl.parameter "v"(0) : %0 // expected-remark{{type: Val}}
    %2 = zhl.global "NondetReg" // expected-remark{{type: NondetReg}}
    %3 = zhl.construct %2(%1) // expected-remark{{type: NondetReg}}
    %4 = zhl.declare "reg" {isPublic = false} // expected-remark{{type: !}}
    zhl.define %4 = %3 // expected-remark{{type: NondetReg}}
    zhl.constrain %1 = %3 // expected-remark{{type: Val}}
    zhl.super %3 // expected-remark{{type: NondetReg}}
  }
  zhl.component @ExtReg {
    %0 = zhl.global "ExtVal" // expected-remark{{type: ExtVal}}
    %1 = zhl.parameter "v"(0) : %0 // expected-remark{{type: ExtVal}}
    %2 = zhl.global "NondetExtReg" // expected-remark{{type: NondetExtReg}}
    %3 = zhl.construct %2(%1) // expected-remark{{type: NondetExtReg}}
    %4 = zhl.declare "reg" {isPublic = false} // expected-remark{{type: !}}
    zhl.define %4 = %3 // expected-remark{{type: NondetExtReg}}
    %5 = zhl.global "EqzExt" // expected-remark{{type: EqzExt}}
    %6 = zhl.global "ExtSub" // expected-remark{{type: ExtSub}}
    %7 = zhl.construct %6(%3, %1) // expected-remark{{type: ExtSub}}
    %8 = zhl.construct %5(%7) // expected-remark{{type: EqzExt}}
    zhl.super %3 // expected-remark{{type: NondetExtReg}}
  }
  zhl.component @Div attributes {function} {
    %0 = zhl.global "Val" // expected-remark{{type: Val}}
    %1 = zhl.parameter "lhs"(0) : %0 // expected-remark{{type: Val}}
    %2 = zhl.global "Val" // expected-remark{{type: Val}}
    %3 = zhl.parameter "rhs"(1) : %2 // expected-remark{{type: Val}}
    %4 = zhl.global "Inv" // expected-remark{{type: Inv}}
    %5 = zhl.construct %4(%3) // expected-remark{{type: Inv}}
    %6 = zhl.declare "reciprocal" {isPublic = false} // expected-remark{{type: !}}
    zhl.define %6 = %5 // expected-remark{{type: Inv}}
    %7 = zhl.global "Mul" // expected-remark{{type: Mul}}
    %8 = zhl.construct %7(%5, %3) // expected-remark{{type: Mul}}
    %9 = zhl.literal 1 // expected-remark{{type: 1}}
    zhl.constrain %8 = %9 // expected-remark{{type: Val}}
    %10 = zhl.global "Mul" // expected-remark{{type: Mul}}
    %11 = zhl.construct %10(%5, %1) // expected-remark{{type: Mul}}
    zhl.super %11 // expected-remark{{type: Mul}}
  }
  zhl.component @Log {
    %0 = zhl.global "String" // expected-remark{{type: String}}
    %1 = zhl.parameter "message"(0) : %0 // expected-remark{{type: String}}
    %2 = zhl.global "Val" // expected-remark{{type: Val}}
    %3 = zhl.parameter "vals"(1) : %2 {variadic} // expected-remark{{type: Val...}}
    %4 = zhl.global "Component" // expected-remark{{type: Component}}
    %5 = zhl.extern "Log"(%1, %3) : %4 // expected-remark{{type: Component}}
    zhl.super %5 // expected-remark{{type: Component}}
  }
  zhl.component @Abort {
    %0 = zhl.global "Component" // expected-remark{{type: Component}}
    %1 = zhl.extern "Abort"() : %0 // expected-remark{{type: Component}}
    zhl.super %1 // expected-remark{{type: Component}}
  }
  zhl.component @Assert {
    %0 = zhl.global "Val" // expected-remark{{type: Val}}
    %1 = zhl.parameter "x"(0) : %0 // expected-remark{{type: Val}}
    %2 = zhl.global "String" // expected-remark{{type: String}}
    %3 = zhl.parameter "message"(1) : %2 // expected-remark{{type: String}}
    %4 = zhl.global "Component" // expected-remark{{type: Component}}
    %5 = zhl.extern "Assert"(%1, %3) : %4 // expected-remark{{type: Component}}
    zhl.super %5 // expected-remark{{type: Component}}
  }
  zhl.component @Unpack attributes {generic} {
    %0 = zhl.global "Val" // expected-remark{{type: Val}}
    %1 = zhl.generic "N"(0) : %0 // expected-remark{{type: N}}
    %2 = zhl.global "Val" // expected-remark{{type: Val}}
    %3 = zhl.generic "P"(1) : %2 // expected-remark{{type: P}}
    %4 = zhl.global "Array" // expected-remark{{type: Array}}
    %5 = zhl.global "Val" // expected-remark{{type: Val}}
    %6 = zhl.global "Div" // expected-remark{{type: Div}}
    %7 = zhl.construct %6(%1, %3) // expected-remark{{type: N / P}}
    %8 = zhl.specialize %4<%5, %7> // expected-remark{{type: Array<Val, N / P>}}
    %9 = zhl.parameter "in"(0) : %8 // expected-remark{{type: Array<Val, Aff$0>}}
    %10 = zhl.literal 0 // expected-remark{{type: 0}}
    %11 = zhl.subscript %9[%10] // expected-remark{{type: Val}}
    zhl.super %11 // expected-remark{{type: Val}}
  }
  zhl.component @User1 {
    %0 = zhl.global "Unpack" // expected-remark{{type: Unpack}}
    %1 = zhl.literal 8 // expected-remark{{type: 8}}
    %2 = zhl.literal 2 // expected-remark{{type: 2}}
    %3 = zhl.specialize %0<%1, %2> // expected-remark{{type: Unpack<8, 2, 4>}}
    %4 = zhl.literal 0 // expected-remark{{type: 0}}
    %5 = zhl.literal 0 // expected-remark{{type: 0}}
    %6 = zhl.literal 0 // expected-remark{{type: 0}}
    %7 = zhl.literal 0 // expected-remark{{type: 0}}
    %8 = zhl.array[%4, %5, %6, %7] // expected-remark{{type: Array<Val, 4>}}
    %9 = zhl.construct %3(%8) // expected-remark{{type: Unpack<8, 2, 4>}}
    zhl.super %9 // expected-remark{{type: Unpack<8, 2, 4>}}
  }
}
