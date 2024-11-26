module {
  zmir.split_component @InRange attributes {builtin} {
    zmir.field @"$super" : !zmir.val
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val, %arg2: !zmir.val) -> !zmir.component<@InRange> {
      %0 = zmir.self : !zmir.component<@InRange>
      %1 = zmir.in_range %arg0 <= %arg1 <  %arg2 : !zmir.val
      zmir.write_field %0[@"$super"] = %1 : <@InRange>, !zmir.val
      return %0 : !zmir.component<@InRange>
    }
    func.func nested @constrain(%arg0: !zmir.component<@InRange>, %arg1: !zmir.val, %arg2: !zmir.val, %arg3: !zmir.val) {
      %0 = zmir.read_field %arg0[@"$super"] : !zmir.component<@InRange>, !zmir.val
      return
    }
  }
  zmir.split_component @Component attributes {builtin} {
    func.func nested @compute() -> !zmir.component<@Component> {
      %0 = zmir.self : !zmir.component<@Component>
      return %0 : !zmir.component<@Component>
    }
    func.func nested @constrain(%arg0: !zmir.component<@Component>) {
      return
    }
  }
  zmir.split_component @NondetReg attributes {builtin} {
    zmir.field @reg : !zmir.val
    zmir.field @"$super" : !zmir.val
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@NondetReg> {
      %0 = zmir.self : !zmir.component<@NondetReg>
      zmir.write_field %0[@reg] = %arg0 : <@NondetReg>, !zmir.val
      zmir.write_field %0[@"$super"] = %arg0 : <@NondetReg>, !zmir.val
      return %0 : !zmir.component<@NondetReg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@NondetReg>, %arg1: !zmir.val) {
      %0 = zmir.read_field %arg0[@reg] : !zmir.component<@NondetReg>, !zmir.val
      %1 = zmir.read_field %arg0[@"$super"] : !zmir.component<@NondetReg>, !zmir.val
      return
    }
  }
  zmir.split_component @Val attributes {builtin} {
    zmir.field @"$super" : !zmir.val
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@Val> {
      %0 = zmir.self : !zmir.component<@Val>
      zmir.write_field %0[@"$super"] = %arg0 : <@Val>, !zmir.val
      return %0 : !zmir.component<@Val>
    }
    func.func nested @constrain(%arg0: !zmir.component<@Val>, %arg1: !zmir.val) {
      %0 = zmir.read_field %arg0[@"$super"] : !zmir.component<@Val>, !zmir.val
      return
    }
  }
  zmir.split_component @Reg attributes {name = "Reg"} {
    zmir.field @"$super" : !zmir.component<@NondetReg>
    zmir.field @reg : !zmir.component<@NondetReg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@Reg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %3 = zmir.constructor {builtin} @NondetReg : (!zmir.val) -> !zmir.component<@NondetReg>
      %4 = call_indirect %3(%2) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@NondetReg>
      %5 = zmir.self : !zmir.component<@Reg>
      zmir.write_field %5[@reg] = %4 : <@Reg>, !zmir.component<@NondetReg>
      zmir.write_field %5[@"$super"] = %4 : <@Reg>, !zmir.component<@NondetReg>
      return %5 : !zmir.component<@Reg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@Reg>, %arg1: !zmir.val) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %arg0[@reg] : !zmir.component<@Reg>, !zmir.component<@NondetReg>
      %3 = zmir.read_field %arg0[@"$super"] : !zmir.component<@Reg>, !zmir.component<@NondetReg>
      %4 = builtin.unrealized_conversion_cast %2 : !zmir.component<@NondetReg> to !zmir.pending
      %5 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %6 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      zmir.constrain %5 = %6 : !zmir.val, !zmir.val
      return
    }
  }
  zmir.split_component @Div attributes {function, name = "Div"} {
    zmir.field @"$super" : !zmir.val
    zmir.field @"$temp" : !zmir.val
    zmir.field @reciprocal : !zmir.val
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val) -> !zmir.component<@Div> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      %5 = zmir.inv %4 : !zmir.val
      %6 = zmir.self : !zmir.component<@Div>
      zmir.write_field %6[@reciprocal] = %5 : <@Div>, !zmir.val
      %7 = zmir.mul %5 : !zmir.val, %4 : !zmir.val
      zmir.write_field %6[@"$temp"] = %7 : <@Div>, !zmir.val
      %8 = zmir.literal 1 : !zmir.val
      %9 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %10 = zmir.mul %5 : !zmir.val, %9 : !zmir.val
      zmir.write_field %6[@"$super"] = %10 : <@Div>, !zmir.val
      return %6 : !zmir.component<@Div>
    }
    func.func nested @constrain(%arg0: !zmir.component<@Div>, %arg1: !zmir.val, %arg2: !zmir.val) {
      %0 = zmir.read_field %arg0[@reciprocal] : !zmir.component<@Div>, !zmir.val
      %1 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@Div>, !zmir.val
      %2 = zmir.literal 1 : !zmir.val
      zmir.constrain %1 = %2 : !zmir.val, !zmir.val
      %3 = zmir.read_field %arg0[@"$super"] : !zmir.component<@Div>, !zmir.val
      return
    }
  }
  zmir.split_component @Log attributes {name = "Log"} {
    zmir.field @"$super" : !zmir.component<@Component>
    func.func private @Log$$extern(!zmir.string, !zmir.pending) -> !zmir.component<@Component> attributes {extern}
    func.func nested @compute(%arg0: !zmir.string, %arg1: !zmir.vargs<!zmir.val>) -> !zmir.component<@Log> {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.vargs<!zmir.val> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = call @Log$$extern(%arg0, %1) {writes_into = "$super"} : (!zmir.string, !zmir.pending) -> !zmir.component<@Component>
      %3 = zmir.self : !zmir.component<@Log>
      zmir.write_field %3[@"$super"] = %2 : <@Log>, !zmir.component<@Component>
      return %3 : !zmir.component<@Log>
    }
    func.func nested @constrain(%arg0: !zmir.component<@Log>, %arg1: !zmir.string, %arg2: !zmir.vargs<!zmir.val>) {
      %0 = builtin.unrealized_conversion_cast %arg2 : !zmir.vargs<!zmir.val> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %arg0[@"$super"] : !zmir.component<@Log>, !zmir.component<@Component>
      %3 = call @Log$$extern(%arg1, %1) {writes_into = "$super"} : (!zmir.string, !zmir.pending) -> !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @AssertBit attributes {function, name = "AssertBit"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.val
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@AssertBit> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.literal 1 : !zmir.val
      %3 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %4 = zmir.sub %2 : !zmir.val, %3 : !zmir.val
      %5 = zmir.self : !zmir.component<@AssertBit>
      zmir.write_field %5[@"$temp"] = %4 : <@AssertBit>, !zmir.val
      %6 = zmir.mul %3 : !zmir.val, %4 : !zmir.val
      zmir.write_field %5[@"$temp_0"] = %6 : <@AssertBit>, !zmir.val
      %7 = zmir.literal 0 : !zmir.val
      %8 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %9 = call_indirect %8() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %5[@"$super"] = %9 : <@AssertBit>, !zmir.component<@Component>
      return %5 : !zmir.component<@AssertBit>
    }
    func.func nested @constrain(%arg0: !zmir.component<@AssertBit>, %arg1: !zmir.val) {
      %0 = zmir.literal 1 : !zmir.val
      %1 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@AssertBit>, !zmir.val
      %2 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@AssertBit>, !zmir.val
      %3 = zmir.literal 0 : !zmir.val
      zmir.constrain %2 = %3 : !zmir.val, !zmir.val
      %4 = zmir.read_field %arg0[@"$super"] : !zmir.component<@AssertBit>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @NondetBitReg attributes {name = "NondetBitReg"} {
    zmir.field @"$super" : !zmir.component<@NondetReg>
    zmir.field @"$temp" : !zmir.component<@AssertBit>
    zmir.field @reg : !zmir.component<@NondetReg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@NondetBitReg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %3 = zmir.constructor {builtin} @NondetReg : (!zmir.val) -> !zmir.component<@NondetReg>
      %4 = call_indirect %3(%2) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@NondetReg>
      %5 = builtin.unrealized_conversion_cast %4 : !zmir.component<@NondetReg> to !zmir.pending
      %6 = zmir.self : !zmir.component<@NondetBitReg>
      zmir.write_field %6[@reg] = %4 : <@NondetBitReg>, !zmir.component<@NondetReg>
      %7 = builtin.unrealized_conversion_cast %5 : !zmir.pending to !zmir.val
      %8 = zmir.constructor @AssertBit : (!zmir.val) -> !zmir.component<@AssertBit>
      %9 = call_indirect %8(%7) {writes_into = "$temp"} : (!zmir.val) -> !zmir.component<@AssertBit>
      zmir.write_field %6[@"$temp"] = %9 : <@NondetBitReg>, !zmir.component<@AssertBit>
      zmir.write_field %6[@"$super"] = %4 : <@NondetBitReg>, !zmir.component<@NondetReg>
      return %6 : !zmir.component<@NondetBitReg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@NondetBitReg>, %arg1: !zmir.val) {
      %0 = zmir.read_field %arg0[@reg] : !zmir.component<@NondetBitReg>, !zmir.component<@NondetReg>
      %1 = zmir.read_field %arg0[@"$super"] : !zmir.component<@NondetBitReg>, !zmir.component<@NondetReg>
      %2 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@NondetBitReg>, !zmir.component<@AssertBit>
      return
    }
  }
  zmir.split_component @BitReg attributes {name = "BitReg"} {
    zmir.field @"$super" : !zmir.component<@NondetBitReg>
    zmir.field @reg : !zmir.component<@NondetBitReg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@BitReg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %3 = zmir.constructor @NondetBitReg : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %4 = call_indirect %3(%2) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %5 = zmir.self : !zmir.component<@BitReg>
      zmir.write_field %5[@reg] = %4 : <@BitReg>, !zmir.component<@NondetBitReg>
      zmir.write_field %5[@"$super"] = %4 : <@BitReg>, !zmir.component<@NondetBitReg>
      return %5 : !zmir.component<@BitReg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@BitReg>, %arg1: !zmir.val) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %arg0[@reg] : !zmir.component<@BitReg>, !zmir.component<@NondetBitReg>
      %3 = zmir.read_field %arg0[@"$super"] : !zmir.component<@BitReg>, !zmir.component<@NondetBitReg>
      %4 = builtin.unrealized_conversion_cast %2 : !zmir.component<@NondetBitReg> to !zmir.pending
      %5 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %6 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      zmir.constrain %5 = %6 : !zmir.val, !zmir.val
      return
    }
  }
  zmir.split_component @AssertTwit attributes {function, name = "AssertTwit"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @"$temp_4" : !zmir.val
    zmir.field @"$temp_3" : !zmir.val
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.val
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.val
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@AssertTwit> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.literal 1 : !zmir.val
      %3 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %4 = zmir.sub %2 : !zmir.val, %3 : !zmir.val
      %5 = zmir.self : !zmir.component<@AssertTwit>
      zmir.write_field %5[@"$temp"] = %4 : <@AssertTwit>, !zmir.val
      %6 = zmir.mul %3 : !zmir.val, %4 : !zmir.val
      zmir.write_field %5[@"$temp_0"] = %6 : <@AssertTwit>, !zmir.val
      %7 = zmir.literal 2 : !zmir.val
      %8 = zmir.sub %7 : !zmir.val, %3 : !zmir.val
      zmir.write_field %5[@"$temp_1"] = %8 : <@AssertTwit>, !zmir.val
      %9 = zmir.mul %6 : !zmir.val, %8 : !zmir.val
      zmir.write_field %5[@"$temp_2"] = %9 : <@AssertTwit>, !zmir.val
      %10 = zmir.literal 3 : !zmir.val
      %11 = zmir.sub %10 : !zmir.val, %3 : !zmir.val
      zmir.write_field %5[@"$temp_3"] = %11 : <@AssertTwit>, !zmir.val
      %12 = zmir.mul %9 : !zmir.val, %11 : !zmir.val
      zmir.write_field %5[@"$temp_4"] = %12 : <@AssertTwit>, !zmir.val
      %13 = zmir.literal 0 : !zmir.val
      %14 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %15 = call_indirect %14() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %5[@"$super"] = %15 : <@AssertTwit>, !zmir.component<@Component>
      return %5 : !zmir.component<@AssertTwit>
    }
    func.func nested @constrain(%arg0: !zmir.component<@AssertTwit>, %arg1: !zmir.val) {
      %0 = zmir.literal 1 : !zmir.val
      %1 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@AssertTwit>, !zmir.val
      %2 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@AssertTwit>, !zmir.val
      %3 = zmir.literal 2 : !zmir.val
      %4 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@AssertTwit>, !zmir.val
      %5 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@AssertTwit>, !zmir.val
      %6 = zmir.literal 3 : !zmir.val
      %7 = zmir.read_field %arg0[@"$temp_3"] : !zmir.component<@AssertTwit>, !zmir.val
      %8 = zmir.read_field %arg0[@"$temp_4"] : !zmir.component<@AssertTwit>, !zmir.val
      %9 = zmir.literal 0 : !zmir.val
      zmir.constrain %8 = %9 : !zmir.val, !zmir.val
      %10 = zmir.read_field %arg0[@"$super"] : !zmir.component<@AssertTwit>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @BitAnd attributes {name = "BitAnd"} {
    zmir.field @"$super" : !zmir.component<@Reg>
    zmir.field @"$temp" : !zmir.val
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val) -> !zmir.component<@BitAnd> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %5 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      %6 = zmir.mul %4 : !zmir.val, %5 : !zmir.val
      %7 = zmir.self : !zmir.component<@BitAnd>
      zmir.write_field %7[@"$temp"] = %6 : <@BitAnd>, !zmir.val
      %8 = zmir.constructor @Reg : (!zmir.val) -> !zmir.component<@Reg>
      %9 = call_indirect %8(%6) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@Reg>
      zmir.write_field %7[@"$super"] = %9 : <@BitAnd>, !zmir.component<@Reg>
      return %7 : !zmir.component<@BitAnd>
    }
    func.func nested @constrain(%arg0: !zmir.component<@BitAnd>, %arg1: !zmir.val, %arg2: !zmir.val) {
      %0 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@BitAnd>, !zmir.val
      %1 = zmir.read_field %arg0[@"$super"] : !zmir.component<@BitAnd>, !zmir.component<@Reg>
      return
    }
  }
  zmir.split_component @BitOr attributes {name = "BitOr"} {
    zmir.field @"$super" : !zmir.component<@Reg>
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.val
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.val
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val) -> !zmir.component<@BitOr> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.literal 1 : !zmir.val
      %5 = zmir.literal 1 : !zmir.val
      %6 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %7 = zmir.sub %5 : !zmir.val, %6 : !zmir.val
      %8 = zmir.self : !zmir.component<@BitOr>
      zmir.write_field %8[@"$temp"] = %7 : <@BitOr>, !zmir.val
      %9 = zmir.literal 1 : !zmir.val
      %10 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      %11 = zmir.sub %9 : !zmir.val, %10 : !zmir.val
      zmir.write_field %8[@"$temp_0"] = %11 : <@BitOr>, !zmir.val
      %12 = zmir.mul %7 : !zmir.val, %11 : !zmir.val
      zmir.write_field %8[@"$temp_1"] = %12 : <@BitOr>, !zmir.val
      %13 = zmir.sub %4 : !zmir.val, %12 : !zmir.val
      zmir.write_field %8[@"$temp_2"] = %13 : <@BitOr>, !zmir.val
      %14 = zmir.constructor @Reg : (!zmir.val) -> !zmir.component<@Reg>
      %15 = call_indirect %14(%13) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@Reg>
      zmir.write_field %8[@"$super"] = %15 : <@BitOr>, !zmir.component<@Reg>
      return %8 : !zmir.component<@BitOr>
    }
    func.func nested @constrain(%arg0: !zmir.component<@BitOr>, %arg1: !zmir.val, %arg2: !zmir.val) {
      %0 = zmir.literal 1 : !zmir.val
      %1 = zmir.literal 1 : !zmir.val
      %2 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@BitOr>, !zmir.val
      %3 = zmir.literal 1 : !zmir.val
      %4 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@BitOr>, !zmir.val
      %5 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@BitOr>, !zmir.val
      %6 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@BitOr>, !zmir.val
      %7 = zmir.read_field %arg0[@"$super"] : !zmir.component<@BitOr>, !zmir.component<@Reg>
      return
    }
  }
  zmir.split_component @NondetTwitReg attributes {name = "NondetTwitReg"} {
    zmir.field @"$super" : !zmir.component<@NondetReg>
    zmir.field @"$temp" : !zmir.component<@AssertTwit>
    zmir.field @reg : !zmir.component<@NondetReg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@NondetTwitReg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %3 = zmir.constructor {builtin} @NondetReg : (!zmir.val) -> !zmir.component<@NondetReg>
      %4 = call_indirect %3(%2) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@NondetReg>
      %5 = builtin.unrealized_conversion_cast %4 : !zmir.component<@NondetReg> to !zmir.pending
      %6 = zmir.self : !zmir.component<@NondetTwitReg>
      zmir.write_field %6[@reg] = %4 : <@NondetTwitReg>, !zmir.component<@NondetReg>
      %7 = builtin.unrealized_conversion_cast %5 : !zmir.pending to !zmir.val
      %8 = zmir.constructor @AssertTwit : (!zmir.val) -> !zmir.component<@AssertTwit>
      %9 = call_indirect %8(%7) {writes_into = "$temp"} : (!zmir.val) -> !zmir.component<@AssertTwit>
      zmir.write_field %6[@"$temp"] = %9 : <@NondetTwitReg>, !zmir.component<@AssertTwit>
      zmir.write_field %6[@"$super"] = %4 : <@NondetTwitReg>, !zmir.component<@NondetReg>
      return %6 : !zmir.component<@NondetTwitReg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@NondetTwitReg>, %arg1: !zmir.val) {
      %0 = zmir.read_field %arg0[@reg] : !zmir.component<@NondetTwitReg>, !zmir.component<@NondetReg>
      %1 = zmir.read_field %arg0[@"$super"] : !zmir.component<@NondetTwitReg>, !zmir.component<@NondetReg>
      %2 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@NondetTwitReg>, !zmir.component<@AssertTwit>
      return
    }
  }
  zmir.split_component @NondetFakeTwitReg attributes {name = "NondetFakeTwitReg"} {
    zmir.field @"$super" : !zmir.val
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.component<@Div>
    zmir.field @"$temp_0" : !zmir.component<@BitAnd>
    zmir.field @"$temp" : !zmir.component<@BitAnd>
    zmir.field @reg1 : !zmir.component<@NondetBitReg>
    zmir.field @reg0 : !zmir.component<@NondetBitReg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@NondetFakeTwitReg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.literal 1 : !zmir.val
      %3 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %4 = zmir.constructor @BitAnd : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %5 = call_indirect %4(%3, %2) {writes_into = "$temp"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %6 = zmir.self : !zmir.component<@NondetFakeTwitReg>
      zmir.write_field %6[@"$temp"] = %5 : <@NondetFakeTwitReg>, !zmir.component<@BitAnd>
      %7 = builtin.unrealized_conversion_cast %5 : !zmir.component<@BitAnd> to !zmir.pending
      %8 = builtin.unrealized_conversion_cast %7 : !zmir.pending to !zmir.val
      %9 = zmir.constructor @NondetBitReg : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %10 = call_indirect %9(%8) {writes_into = "reg0"} : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %11 = builtin.unrealized_conversion_cast %10 : !zmir.component<@NondetBitReg> to !zmir.pending
      zmir.write_field %6[@reg0] = %10 : <@NondetFakeTwitReg>, !zmir.component<@NondetBitReg>
      %12 = zmir.literal 2 : !zmir.val
      %13 = call_indirect %4(%3, %12) {writes_into = "$temp_0"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      zmir.write_field %6[@"$temp_0"] = %13 : <@NondetFakeTwitReg>, !zmir.component<@BitAnd>
      %14 = builtin.unrealized_conversion_cast %13 : !zmir.component<@BitAnd> to !zmir.pending
      %15 = zmir.literal 2 : !zmir.val
      %16 = builtin.unrealized_conversion_cast %14 : !zmir.pending to !zmir.val
      %17 = zmir.constructor @Div : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      %18 = call_indirect %17(%16, %15) {writes_into = "$temp_1"} : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      zmir.write_field %6[@"$temp_1"] = %18 : <@NondetFakeTwitReg>, !zmir.component<@Div>
      %19 = builtin.unrealized_conversion_cast %18 : !zmir.component<@Div> to !zmir.pending
      %20 = builtin.unrealized_conversion_cast %19 : !zmir.pending to !zmir.val
      %21 = call_indirect %9(%20) {writes_into = "reg1"} : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %22 = builtin.unrealized_conversion_cast %21 : !zmir.component<@NondetBitReg> to !zmir.pending
      zmir.write_field %6[@reg1] = %21 : <@NondetFakeTwitReg>, !zmir.component<@NondetBitReg>
      %23 = zmir.literal 2 : !zmir.val
      %24 = builtin.unrealized_conversion_cast %22 : !zmir.pending to !zmir.val
      %25 = zmir.mul %24 : !zmir.val, %23 : !zmir.val
      zmir.write_field %6[@"$temp_2"] = %25 : <@NondetFakeTwitReg>, !zmir.val
      %26 = builtin.unrealized_conversion_cast %11 : !zmir.pending to !zmir.val
      %27 = zmir.add %25 : !zmir.val, %26 : !zmir.val
      zmir.write_field %6[@"$super"] = %27 : <@NondetFakeTwitReg>, !zmir.val
      return %6 : !zmir.component<@NondetFakeTwitReg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@NondetFakeTwitReg>, %arg1: !zmir.val) {
      %0 = zmir.literal 1 : !zmir.val
      %1 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@NondetFakeTwitReg>, !zmir.component<@BitAnd>
      %2 = zmir.read_field %arg0[@reg0] : !zmir.component<@NondetFakeTwitReg>, !zmir.component<@NondetBitReg>
      %3 = zmir.literal 2 : !zmir.val
      %4 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@NondetFakeTwitReg>, !zmir.component<@BitAnd>
      %5 = zmir.literal 2 : !zmir.val
      %6 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@NondetFakeTwitReg>, !zmir.component<@Div>
      %7 = zmir.read_field %arg0[@reg1] : !zmir.component<@NondetFakeTwitReg>, !zmir.component<@NondetBitReg>
      %8 = zmir.literal 2 : !zmir.val
      %9 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@NondetFakeTwitReg>, !zmir.val
      %10 = zmir.read_field %arg0[@"$super"] : !zmir.component<@NondetFakeTwitReg>, !zmir.val
      return
    }
  }
  zmir.split_component @TwitReg attributes {name = "TwitReg"} {
    zmir.field @"$super" : !zmir.component<@NondetTwitReg>
    zmir.field @reg : !zmir.component<@NondetTwitReg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@TwitReg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %3 = zmir.constructor @NondetTwitReg : (!zmir.val) -> !zmir.component<@NondetTwitReg>
      %4 = call_indirect %3(%2) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@NondetTwitReg>
      %5 = zmir.self : !zmir.component<@TwitReg>
      zmir.write_field %5[@reg] = %4 : <@TwitReg>, !zmir.component<@NondetTwitReg>
      zmir.write_field %5[@"$super"] = %4 : <@TwitReg>, !zmir.component<@NondetTwitReg>
      return %5 : !zmir.component<@TwitReg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@TwitReg>, %arg1: !zmir.val) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %arg0[@reg] : !zmir.component<@TwitReg>, !zmir.component<@NondetTwitReg>
      %3 = zmir.read_field %arg0[@"$super"] : !zmir.component<@TwitReg>, !zmir.component<@NondetTwitReg>
      %4 = builtin.unrealized_conversion_cast %2 : !zmir.component<@NondetTwitReg> to !zmir.pending
      %5 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %6 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      zmir.constrain %5 = %6 : !zmir.val, !zmir.val
      return
    }
  }
  zmir.split_component @FakeTwitReg attributes {name = "FakeTwitReg"} {
    zmir.field @"$super" : !zmir.component<@NondetFakeTwitReg>
    zmir.field @reg : !zmir.component<@NondetFakeTwitReg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@FakeTwitReg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %3 = zmir.constructor @NondetFakeTwitReg : (!zmir.val) -> !zmir.component<@NondetFakeTwitReg>
      %4 = call_indirect %3(%2) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@NondetFakeTwitReg>
      %5 = zmir.self : !zmir.component<@FakeTwitReg>
      zmir.write_field %5[@reg] = %4 : <@FakeTwitReg>, !zmir.component<@NondetFakeTwitReg>
      zmir.write_field %5[@"$super"] = %4 : <@FakeTwitReg>, !zmir.component<@NondetFakeTwitReg>
      return %5 : !zmir.component<@FakeTwitReg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@FakeTwitReg>, %arg1: !zmir.val) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %arg0[@reg] : !zmir.component<@FakeTwitReg>, !zmir.component<@NondetFakeTwitReg>
      %3 = zmir.read_field %arg0[@"$super"] : !zmir.component<@FakeTwitReg>, !zmir.component<@NondetFakeTwitReg>
      %4 = builtin.unrealized_conversion_cast %2 : !zmir.component<@NondetFakeTwitReg> to !zmir.pending
      %5 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %6 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      zmir.constrain %5 = %6 : !zmir.val, !zmir.val
      return
    }
  }
  zmir.split_component @IsZero attributes {name = "IsZero"} {
    zmir.field @"$super" : !zmir.component<@NondetReg>
    zmir.field @"$temp_5" : !zmir.val
    zmir.field @"$temp_4" : !zmir.val
    zmir.field @"$temp_3" : !zmir.val
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.component<@AssertBit>
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.val
    zmir.field @inv : !zmir.component<@NondetReg>
    zmir.field @isZero : !zmir.component<@NondetReg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@IsZero> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %3 = zmir.isz %2 : !zmir.val
      %4 = zmir.self : !zmir.component<@IsZero>
      zmir.write_field %4[@"$temp"] = %3 : <@IsZero>, !zmir.val
      %5 = zmir.constructor {builtin} @NondetReg : (!zmir.val) -> !zmir.component<@NondetReg>
      %6 = call_indirect %5(%3) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@NondetReg>
      %7 = builtin.unrealized_conversion_cast %6 : !zmir.component<@NondetReg> to !zmir.pending
      zmir.write_field %4[@isZero] = %6 : <@IsZero>, !zmir.component<@NondetReg>
      %8 = zmir.inv %2 : !zmir.val
      zmir.write_field %4[@"$temp_0"] = %8 : <@IsZero>, !zmir.val
      %9 = call_indirect %5(%8) {writes_into = "inv"} : (!zmir.val) -> !zmir.component<@NondetReg>
      %10 = builtin.unrealized_conversion_cast %9 : !zmir.component<@NondetReg> to !zmir.pending
      zmir.write_field %4[@inv] = %9 : <@IsZero>, !zmir.component<@NondetReg>
      %11 = builtin.unrealized_conversion_cast %7 : !zmir.pending to !zmir.val
      %12 = zmir.constructor @AssertBit : (!zmir.val) -> !zmir.component<@AssertBit>
      %13 = call_indirect %12(%11) {writes_into = "$temp_1"} : (!zmir.val) -> !zmir.component<@AssertBit>
      zmir.write_field %4[@"$temp_1"] = %13 : <@IsZero>, !zmir.component<@AssertBit>
      %14 = builtin.unrealized_conversion_cast %10 : !zmir.pending to !zmir.val
      %15 = zmir.mul %2 : !zmir.val, %14 : !zmir.val
      zmir.write_field %4[@"$temp_2"] = %15 : <@IsZero>, !zmir.val
      %16 = zmir.literal 1 : !zmir.val
      %17 = zmir.sub %16 : !zmir.val, %11 : !zmir.val
      zmir.write_field %4[@"$temp_3"] = %17 : <@IsZero>, !zmir.val
      %18 = zmir.mul %11 : !zmir.val, %2 : !zmir.val
      zmir.write_field %4[@"$temp_4"] = %18 : <@IsZero>, !zmir.val
      %19 = zmir.literal 0 : !zmir.val
      %20 = zmir.mul %11 : !zmir.val, %14 : !zmir.val
      zmir.write_field %4[@"$temp_5"] = %20 : <@IsZero>, !zmir.val
      %21 = zmir.literal 0 : !zmir.val
      zmir.write_field %4[@"$super"] = %6 : <@IsZero>, !zmir.component<@NondetReg>
      return %4 : !zmir.component<@IsZero>
    }
    func.func nested @constrain(%arg0: !zmir.component<@IsZero>, %arg1: !zmir.val) {
      %0 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@IsZero>, !zmir.val
      %1 = zmir.read_field %arg0[@isZero] : !zmir.component<@IsZero>, !zmir.component<@NondetReg>
      %2 = zmir.read_field %arg0[@"$super"] : !zmir.component<@IsZero>, !zmir.component<@NondetReg>
      %3 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@IsZero>, !zmir.val
      %4 = zmir.read_field %arg0[@inv] : !zmir.component<@IsZero>, !zmir.component<@NondetReg>
      %5 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@IsZero>, !zmir.component<@AssertBit>
      %6 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@IsZero>, !zmir.val
      %7 = zmir.literal 1 : !zmir.val
      %8 = zmir.read_field %arg0[@"$temp_3"] : !zmir.component<@IsZero>, !zmir.val
      zmir.constrain %6 = %8 : !zmir.val, !zmir.val
      %9 = zmir.read_field %arg0[@"$temp_4"] : !zmir.component<@IsZero>, !zmir.val
      %10 = zmir.literal 0 : !zmir.val
      zmir.constrain %9 = %10 : !zmir.val, !zmir.val
      %11 = zmir.read_field %arg0[@"$temp_5"] : !zmir.component<@IsZero>, !zmir.val
      %12 = zmir.literal 0 : !zmir.val
      zmir.constrain %11 = %12 : !zmir.val, !zmir.val
      return
    }
  }
  zmir.split_component @LookupDelta attributes {name = "LookupDelta"} {
    zmir.field @"$super" : !zmir.pending
    func.func private @LookupDelta$$extern(!zmir.val, !zmir.pending, !zmir.pending) -> !zmir.pending attributes {extern}
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val, %arg2: !zmir.val) -> !zmir.component<@LookupDelta> {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg2 : !zmir.val to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = call @LookupDelta$$extern(%arg0, %1, %3) {writes_into = "$super"} : (!zmir.val, !zmir.pending, !zmir.pending) -> !zmir.pending
      %5 = zmir.self : !zmir.component<@LookupDelta>
      zmir.write_field %5[@"$super"] = %4 : <@LookupDelta>, !zmir.pending
      return %5 : !zmir.component<@LookupDelta>
    }
    func.func nested @constrain(%arg0: !zmir.component<@LookupDelta>, %arg1: !zmir.val, %arg2: !zmir.val, %arg3: !zmir.val) {
      %0 = builtin.unrealized_conversion_cast %arg2 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg3 : !zmir.val to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %arg0[@"$super"] : !zmir.component<@LookupDelta>, !zmir.pending
      %5 = call @LookupDelta$$extern(%arg1, %1, %3) {writes_into = "$super"} : (!zmir.val, !zmir.pending, !zmir.pending) -> !zmir.pending
      return
    }
  }
  zmir.split_component @LookupCurrent attributes {name = "LookupCurrent"} {
    zmir.field @"$super" : !zmir.pending
    func.func private @LookupCurrent$$extern(!zmir.val, !zmir.pending) -> !zmir.pending attributes {extern}
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val) -> !zmir.component<@LookupCurrent> {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = call @LookupCurrent$$extern(%arg0, %1) {writes_into = "$super"} : (!zmir.val, !zmir.pending) -> !zmir.pending
      %3 = zmir.self : !zmir.component<@LookupCurrent>
      zmir.write_field %3[@"$super"] = %2 : <@LookupCurrent>, !zmir.pending
      return %3 : !zmir.component<@LookupCurrent>
    }
    func.func nested @constrain(%arg0: !zmir.component<@LookupCurrent>, %arg1: !zmir.val, %arg2: !zmir.val) {
      %0 = builtin.unrealized_conversion_cast %arg2 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %arg0[@"$super"] : !zmir.component<@LookupCurrent>, !zmir.pending
      %3 = call @LookupCurrent$$extern(%arg1, %1) {writes_into = "$super"} : (!zmir.val, !zmir.pending) -> !zmir.pending
      return
    }
  }
  zmir.split_component @ArgU8 attributes {argument, name = "ArgU8"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @"$temp" : !zmir.component<@LookupDelta>
    zmir.field @val : !zmir.component<@NondetReg>
    zmir.field @count : !zmir.component<@NondetReg>
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val) -> !zmir.component<@ArgU8> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %5 = zmir.constructor {builtin} @NondetReg : (!zmir.val) -> !zmir.component<@NondetReg>
      %6 = call_indirect %5(%4) {writes_into = "count"} : (!zmir.val) -> !zmir.component<@NondetReg>
      %7 = builtin.unrealized_conversion_cast %6 : !zmir.component<@NondetReg> to !zmir.pending
      %8 = zmir.self : !zmir.component<@ArgU8>
      zmir.write_field %8[@count] = %6 : <@ArgU8>, !zmir.component<@NondetReg>
      %9 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      %10 = call_indirect %5(%9) {writes_into = "val"} : (!zmir.val) -> !zmir.component<@NondetReg>
      %11 = builtin.unrealized_conversion_cast %10 : !zmir.component<@NondetReg> to !zmir.pending
      zmir.write_field %8[@val] = %10 : <@ArgU8>, !zmir.component<@NondetReg>
      %12 = zmir.literal 8 : !zmir.val
      %13 = builtin.unrealized_conversion_cast %11 : !zmir.pending to !zmir.val
      %14 = builtin.unrealized_conversion_cast %7 : !zmir.pending to !zmir.val
      %15 = zmir.constructor @LookupDelta : (!zmir.val, !zmir.val, !zmir.val) -> !zmir.component<@LookupDelta>
      %16 = call_indirect %15(%12, %13, %14) {writes_into = "$temp"} : (!zmir.val, !zmir.val, !zmir.val) -> !zmir.component<@LookupDelta>
      zmir.write_field %8[@"$temp"] = %16 : <@ArgU8>, !zmir.component<@LookupDelta>
      %17 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %18 = call_indirect %17() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %8[@"$super"] = %18 : <@ArgU8>, !zmir.component<@Component>
      return %8 : !zmir.component<@ArgU8>
    }
    func.func nested @constrain(%arg0: !zmir.component<@ArgU8>, %arg1: !zmir.val, %arg2: !zmir.val) {
      %0 = zmir.read_field %arg0[@count] : !zmir.component<@ArgU8>, !zmir.component<@NondetReg>
      %1 = zmir.read_field %arg0[@val] : !zmir.component<@ArgU8>, !zmir.component<@NondetReg>
      %2 = zmir.literal 8 : !zmir.val
      %3 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@ArgU8>, !zmir.component<@LookupDelta>
      %4 = zmir.read_field %arg0[@"$super"] : !zmir.component<@ArgU8>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @NondetU8Reg attributes {name = "NondetU8Reg"} {
    zmir.field @"$super" : !zmir.pending
    zmir.field @arg : !zmir.component<@ArgU8>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@NondetU8Reg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.literal 1 : !zmir.val
      %3 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %4 = zmir.constructor @ArgU8 : (!zmir.val, !zmir.val) -> !zmir.component<@ArgU8>
      %5 = call_indirect %4(%2, %3) {writes_into = "arg"} : (!zmir.val, !zmir.val) -> !zmir.component<@ArgU8>
      %6 = builtin.unrealized_conversion_cast %5 : !zmir.component<@ArgU8> to !zmir.pending
      %7 = zmir.self : !zmir.component<@NondetU8Reg>
      zmir.write_field %7[@arg] = %5 : <@NondetU8Reg>, !zmir.component<@ArgU8>
      %8 = zmir.read_field %6[@count] : !zmir.pending, !zmir.pending
      %9 = zmir.literal 1 : !zmir.val
      %10 = zmir.read_field %6[@val] {writes_into = "$super"} : !zmir.pending, !zmir.pending
      zmir.write_field %7[@"$super"] = %10 : <@NondetU8Reg>, !zmir.pending
      return %7 : !zmir.component<@NondetU8Reg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@NondetU8Reg>, %arg1: !zmir.val) {
      %0 = zmir.literal 1 : !zmir.val
      %1 = zmir.read_field %arg0[@arg] : !zmir.component<@NondetU8Reg>, !zmir.component<@ArgU8>
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.component<@ArgU8> to !zmir.pending
      %3 = zmir.read_field %2[@count] : !zmir.pending, !zmir.pending
      %4 = zmir.literal 1 : !zmir.val
      %5 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      zmir.constrain %5 = %4 : !zmir.val, !zmir.val
      %6 = zmir.read_field %arg0[@"$super"] : !zmir.component<@NondetU8Reg>, !zmir.pending
      %7 = zmir.read_field %2[@val] {writes_into = "$super"} : !zmir.pending, !zmir.pending
      return
    }
  }
  zmir.split_component @U8Reg attributes {name = "U8Reg"} {
    zmir.field @"$super" : !zmir.val
    zmir.field @ret : !zmir.component<@NondetU8Reg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@U8Reg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %3 = zmir.constructor @NondetU8Reg : (!zmir.val) -> !zmir.component<@NondetU8Reg>
      %4 = call_indirect %3(%2) {writes_into = "ret"} : (!zmir.val) -> !zmir.component<@NondetU8Reg>
      %5 = zmir.self : !zmir.component<@U8Reg>
      zmir.write_field %5[@ret] = %4 : <@U8Reg>, !zmir.component<@NondetU8Reg>
      zmir.write_field %5[@"$super"] = %arg0 : <@U8Reg>, !zmir.val
      return %5 : !zmir.component<@U8Reg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@U8Reg>, %arg1: !zmir.val) {
      %0 = zmir.read_field %arg0[@"$super"] : !zmir.component<@U8Reg>, !zmir.val
      %1 = builtin.unrealized_conversion_cast %0 : !zmir.val to !zhl.expr
      %2 = builtin.unrealized_conversion_cast %1 : !zhl.expr to !zmir.pending
      %3 = zmir.read_field %arg0[@ret] : !zmir.component<@U8Reg>, !zmir.component<@NondetU8Reg>
      %4 = builtin.unrealized_conversion_cast %3 : !zmir.component<@NondetU8Reg> to !zmir.pending
      %5 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      %6 = builtin.unrealized_conversion_cast %2 : !zmir.pending to !zmir.val
      zmir.constrain %5 = %6 : !zmir.val, !zmir.val
      return
    }
  }
  zmir.split_component @ArgU16 attributes {argument, name = "ArgU16"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @"$temp" : !zmir.component<@LookupDelta>
    zmir.field @val : !zmir.component<@NondetReg>
    zmir.field @count : !zmir.component<@NondetReg>
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val) -> !zmir.component<@ArgU16> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %5 = zmir.constructor {builtin} @NondetReg : (!zmir.val) -> !zmir.component<@NondetReg>
      %6 = call_indirect %5(%4) {writes_into = "count"} : (!zmir.val) -> !zmir.component<@NondetReg>
      %7 = builtin.unrealized_conversion_cast %6 : !zmir.component<@NondetReg> to !zmir.pending
      %8 = zmir.self : !zmir.component<@ArgU16>
      zmir.write_field %8[@count] = %6 : <@ArgU16>, !zmir.component<@NondetReg>
      %9 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      %10 = call_indirect %5(%9) {writes_into = "val"} : (!zmir.val) -> !zmir.component<@NondetReg>
      %11 = builtin.unrealized_conversion_cast %10 : !zmir.component<@NondetReg> to !zmir.pending
      zmir.write_field %8[@val] = %10 : <@ArgU16>, !zmir.component<@NondetReg>
      %12 = zmir.literal 16 : !zmir.val
      %13 = builtin.unrealized_conversion_cast %11 : !zmir.pending to !zmir.val
      %14 = builtin.unrealized_conversion_cast %7 : !zmir.pending to !zmir.val
      %15 = zmir.constructor @LookupDelta : (!zmir.val, !zmir.val, !zmir.val) -> !zmir.component<@LookupDelta>
      %16 = call_indirect %15(%12, %13, %14) {writes_into = "$temp"} : (!zmir.val, !zmir.val, !zmir.val) -> !zmir.component<@LookupDelta>
      zmir.write_field %8[@"$temp"] = %16 : <@ArgU16>, !zmir.component<@LookupDelta>
      %17 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %18 = call_indirect %17() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %8[@"$super"] = %18 : <@ArgU16>, !zmir.component<@Component>
      return %8 : !zmir.component<@ArgU16>
    }
    func.func nested @constrain(%arg0: !zmir.component<@ArgU16>, %arg1: !zmir.val, %arg2: !zmir.val) {
      %0 = zmir.read_field %arg0[@count] : !zmir.component<@ArgU16>, !zmir.component<@NondetReg>
      %1 = zmir.read_field %arg0[@val] : !zmir.component<@ArgU16>, !zmir.component<@NondetReg>
      %2 = zmir.literal 16 : !zmir.val
      %3 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@ArgU16>, !zmir.component<@LookupDelta>
      %4 = zmir.read_field %arg0[@"$super"] : !zmir.component<@ArgU16>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @NondetU16Reg attributes {name = "NondetU16Reg"} {
    zmir.field @"$super" : !zmir.pending
    zmir.field @arg : !zmir.component<@ArgU16>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@NondetU16Reg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.literal 1 : !zmir.val
      %3 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %4 = zmir.constructor @ArgU16 : (!zmir.val, !zmir.val) -> !zmir.component<@ArgU16>
      %5 = call_indirect %4(%2, %3) {writes_into = "arg"} : (!zmir.val, !zmir.val) -> !zmir.component<@ArgU16>
      %6 = builtin.unrealized_conversion_cast %5 : !zmir.component<@ArgU16> to !zmir.pending
      %7 = zmir.self : !zmir.component<@NondetU16Reg>
      zmir.write_field %7[@arg] = %5 : <@NondetU16Reg>, !zmir.component<@ArgU16>
      %8 = zmir.read_field %6[@count] : !zmir.pending, !zmir.pending
      %9 = zmir.literal 1 : !zmir.val
      %10 = zmir.read_field %6[@val] {writes_into = "$super"} : !zmir.pending, !zmir.pending
      zmir.write_field %7[@"$super"] = %10 : <@NondetU16Reg>, !zmir.pending
      return %7 : !zmir.component<@NondetU16Reg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@NondetU16Reg>, %arg1: !zmir.val) {
      %0 = zmir.literal 1 : !zmir.val
      %1 = zmir.read_field %arg0[@arg] : !zmir.component<@NondetU16Reg>, !zmir.component<@ArgU16>
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.component<@ArgU16> to !zmir.pending
      %3 = zmir.read_field %2[@count] : !zmir.pending, !zmir.pending
      %4 = zmir.literal 1 : !zmir.val
      %5 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      zmir.constrain %5 = %4 : !zmir.val, !zmir.val
      %6 = zmir.read_field %arg0[@"$super"] : !zmir.component<@NondetU16Reg>, !zmir.pending
      %7 = zmir.read_field %2[@val] {writes_into = "$super"} : !zmir.pending, !zmir.pending
      return
    }
  }
  zmir.split_component @U16Reg attributes {name = "U16Reg"} {
    zmir.field @"$super" : !zmir.val
    zmir.field @ret : !zmir.component<@NondetU16Reg>
    func.func nested @compute(%arg0: !zmir.val) -> !zmir.component<@U16Reg> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %3 = zmir.constructor @NondetU16Reg : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %4 = call_indirect %3(%2) {writes_into = "ret"} : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %5 = zmir.self : !zmir.component<@U16Reg>
      zmir.write_field %5[@ret] = %4 : <@U16Reg>, !zmir.component<@NondetU16Reg>
      zmir.write_field %5[@"$super"] = %arg0 : <@U16Reg>, !zmir.val
      return %5 : !zmir.component<@U16Reg>
    }
    func.func nested @constrain(%arg0: !zmir.component<@U16Reg>, %arg1: !zmir.val) {
      %0 = zmir.read_field %arg0[@"$super"] : !zmir.component<@U16Reg>, !zmir.val
      %1 = builtin.unrealized_conversion_cast %0 : !zmir.val to !zhl.expr
      %2 = builtin.unrealized_conversion_cast %1 : !zhl.expr to !zmir.pending
      %3 = zmir.read_field %arg0[@ret] : !zmir.component<@U16Reg>, !zmir.component<@NondetU16Reg>
      %4 = builtin.unrealized_conversion_cast %3 : !zmir.component<@NondetU16Reg> to !zmir.pending
      %5 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      %6 = builtin.unrealized_conversion_cast %2 : !zmir.pending to !zmir.val
      zmir.constrain %5 = %6 : !zmir.val, !zmir.val
      return
    }
  }
  zmir.split_component @ValU32 attributes {name = "ValU32"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @high : !zmir.val
    zmir.field @low : !zmir.val
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val) -> !zmir.component<@ValU32> {
      %0 = zmir.self : !zmir.component<@ValU32>
      zmir.write_field %0[@low] = %arg0 : <@ValU32>, !zmir.val
      zmir.write_field %0[@high] = %arg1 : <@ValU32>, !zmir.val
      %1 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %2 = call_indirect %1() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %0[@"$super"] = %2 : <@ValU32>, !zmir.component<@Component>
      return %0 : !zmir.component<@ValU32>
    }
    func.func nested @constrain(%arg0: !zmir.component<@ValU32>, %arg1: !zmir.val, %arg2: !zmir.val) {
      %0 = zmir.read_field %arg0[@high] : !zmir.component<@ValU32>, !zmir.val
      %1 = zmir.read_field %arg0[@low] : !zmir.component<@ValU32>, !zmir.val
      %2 = zmir.read_field %arg0[@"$super"] : !zmir.component<@ValU32>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @DenormedValU32 attributes {name = "DenormedValU32"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @high : !zmir.val
    zmir.field @low : !zmir.val
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.val) -> !zmir.component<@DenormedValU32> {
      %0 = zmir.self : !zmir.component<@DenormedValU32>
      zmir.write_field %0[@low] = %arg0 : <@DenormedValU32>, !zmir.val
      zmir.write_field %0[@high] = %arg1 : <@DenormedValU32>, !zmir.val
      %1 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %2 = call_indirect %1() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %0[@"$super"] = %2 : <@DenormedValU32>, !zmir.component<@Component>
      return %0 : !zmir.component<@DenormedValU32>
    }
    func.func nested @constrain(%arg0: !zmir.component<@DenormedValU32>, %arg1: !zmir.val, %arg2: !zmir.val) {
      %0 = zmir.read_field %arg0[@high] : !zmir.component<@DenormedValU32>, !zmir.val
      %1 = zmir.read_field %arg0[@low] : !zmir.component<@DenormedValU32>, !zmir.val
      %2 = zmir.read_field %arg0[@"$super"] : !zmir.component<@DenormedValU32>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @AddU32 attributes {name = "AddU32"} {
    zmir.field @"$super" : !zmir.component<@DenormedValU32>
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.val
    func.func nested @compute(%arg0: !zmir.component<@ValU32>, %arg1: !zmir.component<@ValU32>) -> !zmir.component<@AddU32> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %5 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %6 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      %7 = builtin.unrealized_conversion_cast %5 : !zmir.pending to !zmir.val
      %8 = zmir.add %6 : !zmir.val, %7 : !zmir.val
      %9 = zmir.self : !zmir.component<@AddU32>
      zmir.write_field %9[@"$temp"] = %8 : <@AddU32>, !zmir.val
      %10 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %11 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %12 = builtin.unrealized_conversion_cast %10 : !zmir.pending to !zmir.val
      %13 = builtin.unrealized_conversion_cast %11 : !zmir.pending to !zmir.val
      %14 = zmir.add %12 : !zmir.val, %13 : !zmir.val
      zmir.write_field %9[@"$temp_0"] = %14 : <@AddU32>, !zmir.val
      %15 = zmir.constructor @DenormedValU32 : (!zmir.val, !zmir.val) -> !zmir.component<@DenormedValU32>
      %16 = call_indirect %15(%8, %14) {writes_into = "$super"} : (!zmir.val, !zmir.val) -> !zmir.component<@DenormedValU32>
      zmir.write_field %9[@"$super"] = %16 : <@AddU32>, !zmir.component<@DenormedValU32>
      return %9 : !zmir.component<@AddU32>
    }
    func.func nested @constrain(%arg0: !zmir.component<@AddU32>, %arg1: !zmir.component<@ValU32>, %arg2: !zmir.component<@ValU32>) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg2 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %5 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %6 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@AddU32>, !zmir.val
      %7 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %8 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %9 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@AddU32>, !zmir.val
      %10 = zmir.read_field %arg0[@"$super"] : !zmir.component<@AddU32>, !zmir.component<@DenormedValU32>
      return
    }
  }
  zmir.split_component @SubU32 attributes {name = "SubU32"} {
    zmir.field @"$super" : !zmir.component<@DenormedValU32>
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.val
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.val
    func.func nested @compute(%arg0: !zmir.component<@ValU32>, %arg1: !zmir.component<@ValU32>) -> !zmir.component<@SubU32> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.literal 65536 : !zmir.val
      %5 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %6 = builtin.unrealized_conversion_cast %5 : !zmir.pending to !zmir.val
      %7 = zmir.add %4 : !zmir.val, %6 : !zmir.val
      %8 = zmir.self : !zmir.component<@SubU32>
      zmir.write_field %8[@"$temp"] = %7 : <@SubU32>, !zmir.val
      %9 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %10 = builtin.unrealized_conversion_cast %9 : !zmir.pending to !zmir.val
      %11 = zmir.sub %7 : !zmir.val, %10 : !zmir.val
      zmir.write_field %8[@"$temp_0"] = %11 : <@SubU32>, !zmir.val
      %12 = zmir.literal 65535 : !zmir.val
      %13 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %14 = builtin.unrealized_conversion_cast %13 : !zmir.pending to !zmir.val
      %15 = zmir.add %12 : !zmir.val, %14 : !zmir.val
      zmir.write_field %8[@"$temp_1"] = %15 : <@SubU32>, !zmir.val
      %16 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %17 = builtin.unrealized_conversion_cast %16 : !zmir.pending to !zmir.val
      %18 = zmir.sub %15 : !zmir.val, %17 : !zmir.val
      zmir.write_field %8[@"$temp_2"] = %18 : <@SubU32>, !zmir.val
      %19 = zmir.constructor @DenormedValU32 : (!zmir.val, !zmir.val) -> !zmir.component<@DenormedValU32>
      %20 = call_indirect %19(%11, %18) {writes_into = "$super"} : (!zmir.val, !zmir.val) -> !zmir.component<@DenormedValU32>
      zmir.write_field %8[@"$super"] = %20 : <@SubU32>, !zmir.component<@DenormedValU32>
      return %8 : !zmir.component<@SubU32>
    }
    func.func nested @constrain(%arg0: !zmir.component<@SubU32>, %arg1: !zmir.component<@ValU32>, %arg2: !zmir.component<@ValU32>) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg2 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.literal 65536 : !zmir.val
      %5 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %6 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@SubU32>, !zmir.val
      %7 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %8 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@SubU32>, !zmir.val
      %9 = zmir.literal 65535 : !zmir.val
      %10 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %11 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@SubU32>, !zmir.val
      %12 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %13 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@SubU32>, !zmir.val
      %14 = zmir.read_field %arg0[@"$super"] : !zmir.component<@SubU32>, !zmir.component<@DenormedValU32>
      return
    }
  }
  zmir.split_component @Denorm attributes {name = "Denorm"} {
    zmir.field @"$super" : !zmir.component<@DenormedValU32>
    func.func nested @compute(%arg0: !zmir.component<@ValU32>) -> !zmir.component<@Denorm> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %3 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %4 = builtin.unrealized_conversion_cast %2 : !zmir.pending to !zmir.val
      %5 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      %6 = zmir.constructor @DenormedValU32 : (!zmir.val, !zmir.val) -> !zmir.component<@DenormedValU32>
      %7 = call_indirect %6(%4, %5) {writes_into = "$super"} : (!zmir.val, !zmir.val) -> !zmir.component<@DenormedValU32>
      %8 = zmir.self : !zmir.component<@Denorm>
      zmir.write_field %8[@"$super"] = %7 : <@Denorm>, !zmir.component<@DenormedValU32>
      return %8 : !zmir.component<@Denorm>
    }
    func.func nested @constrain(%arg0: !zmir.component<@Denorm>, %arg1: !zmir.component<@ValU32>) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %3 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %4 = zmir.read_field %arg0[@"$super"] : !zmir.component<@Denorm>, !zmir.component<@DenormedValU32>
      return
    }
  }
  zmir.split_component @NormalizeU32 attributes {name = "NormalizeU32"} {
    zmir.field @"$super" : !zmir.component<@ValU32>
    zmir.field @"$temp_8" : !zmir.val
    zmir.field @"$temp_7" : !zmir.val
    zmir.field @"$temp_6" : !zmir.component<@Div>
    zmir.field @"$temp_5" : !zmir.component<@BitAnd>
    zmir.field @"$temp_4" : !zmir.component<@BitAnd>
    zmir.field @"$temp_3" : !zmir.val
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.component<@Div>
    zmir.field @"$temp_0" : !zmir.component<@BitAnd>
    zmir.field @"$temp" : !zmir.component<@BitAnd>
    zmir.field @carry : !zmir.component<@NondetBitReg>
    zmir.field @highCarry : !zmir.component<@NondetBitReg>
    zmir.field @high16 : !zmir.component<@NondetU16Reg>
    zmir.field @high : !zmir.val
    zmir.field @lowCarry : !zmir.component<@NondetBitReg>
    zmir.field @low16 : !zmir.component<@NondetU16Reg>
    func.func nested @compute(%arg0: !zmir.component<@DenormedValU32>) -> !zmir.component<@NormalizeU32> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@DenormedValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %3 = zmir.literal 65535 : !zmir.val
      %4 = builtin.unrealized_conversion_cast %2 : !zmir.pending to !zmir.val
      %5 = zmir.constructor @BitAnd : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %6 = call_indirect %5(%4, %3) {writes_into = "$temp"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %7 = zmir.self : !zmir.component<@NormalizeU32>
      zmir.write_field %7[@"$temp"] = %6 : <@NormalizeU32>, !zmir.component<@BitAnd>
      %8 = builtin.unrealized_conversion_cast %6 : !zmir.component<@BitAnd> to !zmir.pending
      %9 = builtin.unrealized_conversion_cast %8 : !zmir.pending to !zmir.val
      %10 = zmir.constructor @NondetU16Reg : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %11 = call_indirect %10(%9) {writes_into = "low16"} : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %12 = builtin.unrealized_conversion_cast %11 : !zmir.component<@NondetU16Reg> to !zmir.pending
      zmir.write_field %7[@low16] = %11 : <@NormalizeU32>, !zmir.component<@NondetU16Reg>
      %13 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %14 = zmir.literal 65536 : !zmir.val
      %15 = builtin.unrealized_conversion_cast %13 : !zmir.pending to !zmir.val
      %16 = call_indirect %5(%15, %14) {writes_into = "$temp_0"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      zmir.write_field %7[@"$temp_0"] = %16 : <@NormalizeU32>, !zmir.component<@BitAnd>
      %17 = builtin.unrealized_conversion_cast %16 : !zmir.component<@BitAnd> to !zmir.pending
      %18 = zmir.literal 65536 : !zmir.val
      %19 = builtin.unrealized_conversion_cast %17 : !zmir.pending to !zmir.val
      %20 = zmir.constructor @Div : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      %21 = call_indirect %20(%19, %18) {writes_into = "$temp_1"} : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      zmir.write_field %7[@"$temp_1"] = %21 : <@NormalizeU32>, !zmir.component<@Div>
      %22 = builtin.unrealized_conversion_cast %21 : !zmir.component<@Div> to !zmir.pending
      %23 = builtin.unrealized_conversion_cast %22 : !zmir.pending to !zmir.val
      %24 = zmir.constructor @NondetBitReg : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %25 = call_indirect %24(%23) {writes_into = "lowCarry"} : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %26 = builtin.unrealized_conversion_cast %25 : !zmir.component<@NondetBitReg> to !zmir.pending
      zmir.write_field %7[@lowCarry] = %25 : <@NormalizeU32>, !zmir.component<@NondetBitReg>
      %27 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %28 = zmir.literal 65536 : !zmir.val
      %29 = builtin.unrealized_conversion_cast %26 : !zmir.pending to !zmir.val
      %30 = zmir.mul %29 : !zmir.val, %28 : !zmir.val
      zmir.write_field %7[@"$temp_2"] = %30 : <@NormalizeU32>, !zmir.val
      %31 = builtin.unrealized_conversion_cast %12 : !zmir.pending to !zmir.val
      %32 = zmir.add %30 : !zmir.val, %31 : !zmir.val
      zmir.write_field %7[@"$temp_3"] = %32 : <@NormalizeU32>, !zmir.val
      %33 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %34 = builtin.unrealized_conversion_cast %33 : !zmir.pending to !zmir.val
      %35 = zmir.add %34 : !zmir.val, %29 : !zmir.val
      zmir.write_field %7[@high] = %35 : <@NormalizeU32>, !zmir.val
      %36 = zmir.literal 65535 : !zmir.val
      %37 = call_indirect %5(%35, %36) {writes_into = "$temp_4"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      zmir.write_field %7[@"$temp_4"] = %37 : <@NormalizeU32>, !zmir.component<@BitAnd>
      %38 = builtin.unrealized_conversion_cast %37 : !zmir.component<@BitAnd> to !zmir.pending
      %39 = builtin.unrealized_conversion_cast %38 : !zmir.pending to !zmir.val
      %40 = call_indirect %10(%39) {writes_into = "high16"} : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %41 = builtin.unrealized_conversion_cast %40 : !zmir.component<@NondetU16Reg> to !zmir.pending
      zmir.write_field %7[@high16] = %40 : <@NormalizeU32>, !zmir.component<@NondetU16Reg>
      %42 = zmir.literal 65536 : !zmir.val
      %43 = call_indirect %5(%35, %42) {writes_into = "$temp_5"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      zmir.write_field %7[@"$temp_5"] = %43 : <@NormalizeU32>, !zmir.component<@BitAnd>
      %44 = builtin.unrealized_conversion_cast %43 : !zmir.component<@BitAnd> to !zmir.pending
      %45 = zmir.literal 65536 : !zmir.val
      %46 = builtin.unrealized_conversion_cast %44 : !zmir.pending to !zmir.val
      %47 = call_indirect %20(%46, %45) {writes_into = "$temp_6"} : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      zmir.write_field %7[@"$temp_6"] = %47 : <@NormalizeU32>, !zmir.component<@Div>
      %48 = builtin.unrealized_conversion_cast %47 : !zmir.component<@Div> to !zmir.pending
      %49 = builtin.unrealized_conversion_cast %48 : !zmir.pending to !zmir.val
      %50 = call_indirect %24(%49) {writes_into = "carry"} : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %51 = builtin.unrealized_conversion_cast %50 : !zmir.component<@NondetBitReg> to !zmir.pending
      zmir.write_field %7[@highCarry] = %50 : <@NormalizeU32>, !zmir.component<@NondetBitReg>
      %52 = zmir.literal 65536 : !zmir.val
      %53 = builtin.unrealized_conversion_cast %51 : !zmir.pending to !zmir.val
      %54 = zmir.mul %53 : !zmir.val, %52 : !zmir.val
      zmir.write_field %7[@"$temp_7"] = %54 : <@NormalizeU32>, !zmir.val
      %55 = builtin.unrealized_conversion_cast %41 : !zmir.pending to !zmir.val
      %56 = zmir.add %54 : !zmir.val, %55 : !zmir.val
      zmir.write_field %7[@"$temp_8"] = %56 : <@NormalizeU32>, !zmir.val
      zmir.write_field %7[@carry] = %50 : <@NormalizeU32>, !zmir.component<@NondetBitReg>
      %57 = zmir.constructor @ValU32 : (!zmir.val, !zmir.val) -> !zmir.component<@ValU32>
      %58 = call_indirect %57(%31, %55) {writes_into = "$super"} : (!zmir.val, !zmir.val) -> !zmir.component<@ValU32>
      zmir.write_field %7[@"$super"] = %58 : <@NormalizeU32>, !zmir.component<@ValU32>
      return %7 : !zmir.component<@NormalizeU32>
    }
    func.func nested @constrain(%arg0: !zmir.component<@NormalizeU32>, %arg1: !zmir.component<@DenormedValU32>) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@DenormedValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %3 = zmir.literal 65535 : !zmir.val
      %4 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@NormalizeU32>, !zmir.component<@BitAnd>
      %5 = zmir.read_field %arg0[@low16] : !zmir.component<@NormalizeU32>, !zmir.component<@NondetU16Reg>
      %6 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %7 = zmir.literal 65536 : !zmir.val
      %8 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@NormalizeU32>, !zmir.component<@BitAnd>
      %9 = zmir.literal 65536 : !zmir.val
      %10 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@NormalizeU32>, !zmir.component<@Div>
      %11 = zmir.read_field %arg0[@lowCarry] : !zmir.component<@NormalizeU32>, !zmir.component<@NondetBitReg>
      %12 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %13 = zmir.literal 65536 : !zmir.val
      %14 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@NormalizeU32>, !zmir.val
      %15 = zmir.read_field %arg0[@"$temp_3"] : !zmir.component<@NormalizeU32>, !zmir.val
      %16 = builtin.unrealized_conversion_cast %12 : !zmir.pending to !zmir.val
      zmir.constrain %16 = %15 : !zmir.val, !zmir.val
      %17 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %18 = zmir.read_field %arg0[@high] : !zmir.component<@NormalizeU32>, !zmir.val
      %19 = zmir.literal 65535 : !zmir.val
      %20 = zmir.read_field %arg0[@"$temp_4"] : !zmir.component<@NormalizeU32>, !zmir.component<@BitAnd>
      %21 = zmir.read_field %arg0[@high16] : !zmir.component<@NormalizeU32>, !zmir.component<@NondetU16Reg>
      %22 = zmir.literal 65536 : !zmir.val
      %23 = zmir.read_field %arg0[@"$temp_5"] : !zmir.component<@NormalizeU32>, !zmir.component<@BitAnd>
      %24 = zmir.literal 65536 : !zmir.val
      %25 = zmir.read_field %arg0[@"$temp_6"] : !zmir.component<@NormalizeU32>, !zmir.component<@Div>
      %26 = zmir.read_field %arg0[@highCarry] : !zmir.component<@NormalizeU32>, !zmir.component<@NondetBitReg>
      %27 = zmir.read_field %arg0[@carry] : !zmir.component<@NormalizeU32>, !zmir.component<@NondetBitReg>
      %28 = zmir.literal 65536 : !zmir.val
      %29 = zmir.read_field %arg0[@"$temp_7"] : !zmir.component<@NormalizeU32>, !zmir.val
      %30 = zmir.read_field %arg0[@"$temp_8"] : !zmir.component<@NormalizeU32>, !zmir.val
      zmir.constrain %18 = %30 : !zmir.val, !zmir.val
      %31 = zmir.read_field %arg0[@"$super"] : !zmir.component<@NormalizeU32>, !zmir.component<@ValU32>
      return
    }
  }
  zmir.split_component @AddrDecompose attributes {name = "AddrDecompose"} {
    zmir.field @"$super" : !zmir.val
    zmir.field @"$temp_10" : !zmir.val
    zmir.field @"$temp_9" : !zmir.val
    zmir.field @"$temp_8" : !zmir.val
    zmir.field @"$temp_7" : !zmir.component<@Div>
    zmir.field @"$temp_6" : !zmir.val
    zmir.field @"$temp_5" : !zmir.component<@IsZero>
    zmir.field @"$temp_4" : !zmir.val
    zmir.field @"$temp_3" : !zmir.val
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.val
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.component<@BitAnd>
    zmir.field @addr : !zmir.val
    zmir.field @med14 : !zmir.component<@NondetU16Reg>
    zmir.field @upperDiff : !zmir.component<@U16Reg>
    zmir.field @low2 : !zmir.component<@NondetTwitReg>
    func.func nested @compute(%arg0: !zmir.component<@ValU32>, %arg1: !zmir.val) -> !zmir.component<@AddrDecompose> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %5 = zmir.literal 3 : !zmir.val
      %6 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      %7 = zmir.constructor @BitAnd : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %8 = call_indirect %7(%6, %5) {writes_into = "$temp"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %9 = zmir.self : !zmir.component<@AddrDecompose>
      zmir.write_field %9[@"$temp"] = %8 : <@AddrDecompose>, !zmir.component<@BitAnd>
      %10 = builtin.unrealized_conversion_cast %8 : !zmir.component<@BitAnd> to !zmir.pending
      %11 = builtin.unrealized_conversion_cast %10 : !zmir.pending to !zmir.val
      %12 = zmir.constructor @NondetTwitReg : (!zmir.val) -> !zmir.component<@NondetTwitReg>
      %13 = call_indirect %12(%11) {writes_into = "low2"} : (!zmir.val) -> !zmir.component<@NondetTwitReg>
      %14 = builtin.unrealized_conversion_cast %13 : !zmir.component<@NondetTwitReg> to !zmir.pending
      zmir.write_field %9[@low2] = %13 : <@AddrDecompose>, !zmir.component<@NondetTwitReg>
      %15 = zmir.literal 65535 : !zmir.val
      %16 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      %17 = zmir.mul %16 : !zmir.val, %15 : !zmir.val
      zmir.write_field %9[@"$temp_0"] = %17 : <@AddrDecompose>, !zmir.val
      %18 = zmir.literal 1 : !zmir.val
      %19 = zmir.sub %18 : !zmir.val, %16 : !zmir.val
      zmir.write_field %9[@"$temp_1"] = %19 : <@AddrDecompose>, !zmir.val
      %20 = zmir.literal 49151 : !zmir.val
      %21 = zmir.mul %19 : !zmir.val, %20 : !zmir.val
      zmir.write_field %9[@"$temp_2"] = %21 : <@AddrDecompose>, !zmir.val
      %22 = zmir.add %17 : !zmir.val, %21 : !zmir.val
      zmir.write_field %9[@"$temp_3"] = %22 : <@AddrDecompose>, !zmir.val
      %23 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %24 = builtin.unrealized_conversion_cast %23 : !zmir.pending to !zmir.val
      %25 = zmir.sub %22 : !zmir.val, %24 : !zmir.val
      zmir.write_field %9[@"$temp_4"] = %25 : <@AddrDecompose>, !zmir.val
      %26 = zmir.constructor @U16Reg : (!zmir.val) -> !zmir.component<@U16Reg>
      %27 = call_indirect %26(%25) {writes_into = "upperDiff"} : (!zmir.val) -> !zmir.component<@U16Reg>
      zmir.write_field %9[@upperDiff] = %27 : <@AddrDecompose>, !zmir.component<@U16Reg>
      %28 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %29 = builtin.unrealized_conversion_cast %28 : !zmir.pending to !zmir.val
      %30 = zmir.constructor @IsZero : (!zmir.val) -> !zmir.component<@IsZero>
      %31 = call_indirect %30(%29) {writes_into = "$temp_5"} : (!zmir.val) -> !zmir.component<@IsZero>
      zmir.write_field %9[@"$temp_5"] = %31 : <@AddrDecompose>, !zmir.component<@IsZero>
      %32 = zmir.literal 0 : !zmir.val
      %33 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %34 = builtin.unrealized_conversion_cast %33 : !zmir.pending to !zmir.val
      %35 = builtin.unrealized_conversion_cast %14 : !zmir.pending to !zmir.val
      %36 = zmir.sub %34 : !zmir.val, %35 : !zmir.val
      zmir.write_field %9[@"$temp_6"] = %36 : <@AddrDecompose>, !zmir.val
      %37 = zmir.literal 4 : !zmir.val
      %38 = zmir.constructor @Div : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      %39 = call_indirect %38(%36, %37) {writes_into = "$temp_7"} : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      zmir.write_field %9[@"$temp_7"] = %39 : <@AddrDecompose>, !zmir.component<@Div>
      %40 = builtin.unrealized_conversion_cast %39 : !zmir.component<@Div> to !zmir.pending
      %41 = builtin.unrealized_conversion_cast %40 : !zmir.pending to !zmir.val
      %42 = zmir.constructor @NondetU16Reg : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %43 = call_indirect %42(%41) {writes_into = "med14"} : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %44 = builtin.unrealized_conversion_cast %43 : !zmir.component<@NondetU16Reg> to !zmir.pending
      zmir.write_field %9[@med14] = %43 : <@AddrDecompose>, !zmir.component<@NondetU16Reg>
      %45 = zmir.literal 4 : !zmir.val
      %46 = builtin.unrealized_conversion_cast %44 : !zmir.pending to !zmir.val
      %47 = zmir.mul %46 : !zmir.val, %45 : !zmir.val
      zmir.write_field %9[@"$temp_8"] = %47 : <@AddrDecompose>, !zmir.val
      %48 = zmir.add %47 : !zmir.val, %35 : !zmir.val
      zmir.write_field %9[@"$temp_9"] = %48 : <@AddrDecompose>, !zmir.val
      %49 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %50 = zmir.literal 16384 : !zmir.val
      %51 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %52 = builtin.unrealized_conversion_cast %51 : !zmir.pending to !zmir.val
      %53 = zmir.mul %50 : !zmir.val, %52 : !zmir.val
      zmir.write_field %9[@"$temp_10"] = %53 : <@AddrDecompose>, !zmir.val
      %54 = zmir.add %53 : !zmir.val, %46 : !zmir.val
      zmir.write_field %9[@addr] = %54 : <@AddrDecompose>, !zmir.val
      zmir.write_field %9[@"$super"] = %54 : <@AddrDecompose>, !zmir.val
      return %9 : !zmir.component<@AddrDecompose>
    }
    func.func nested @constrain(%arg0: !zmir.component<@AddrDecompose>, %arg1: !zmir.component<@ValU32>, %arg2: !zmir.val) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %3 = zmir.literal 3 : !zmir.val
      %4 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@AddrDecompose>, !zmir.component<@BitAnd>
      %5 = zmir.read_field %arg0[@low2] : !zmir.component<@AddrDecompose>, !zmir.component<@NondetTwitReg>
      %6 = zmir.literal 65535 : !zmir.val
      %7 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@AddrDecompose>, !zmir.val
      %8 = zmir.literal 1 : !zmir.val
      %9 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@AddrDecompose>, !zmir.val
      %10 = zmir.literal 49151 : !zmir.val
      %11 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@AddrDecompose>, !zmir.val
      %12 = zmir.read_field %arg0[@"$temp_3"] : !zmir.component<@AddrDecompose>, !zmir.val
      %13 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %14 = zmir.read_field %arg0[@"$temp_4"] : !zmir.component<@AddrDecompose>, !zmir.val
      %15 = zmir.read_field %arg0[@upperDiff] : !zmir.component<@AddrDecompose>, !zmir.component<@U16Reg>
      %16 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %17 = zmir.read_field %arg0[@"$temp_5"] : !zmir.component<@AddrDecompose>, !zmir.component<@IsZero>
      %18 = builtin.unrealized_conversion_cast %17 : !zmir.component<@IsZero> to !zmir.pending
      %19 = zmir.literal 0 : !zmir.val
      %20 = builtin.unrealized_conversion_cast %18 : !zmir.pending to !zmir.val
      zmir.constrain %20 = %19 : !zmir.val, !zmir.val
      %21 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %22 = zmir.read_field %arg0[@"$temp_6"] : !zmir.component<@AddrDecompose>, !zmir.val
      %23 = zmir.literal 4 : !zmir.val
      %24 = zmir.read_field %arg0[@"$temp_7"] : !zmir.component<@AddrDecompose>, !zmir.component<@Div>
      %25 = zmir.read_field %arg0[@med14] : !zmir.component<@AddrDecompose>, !zmir.component<@NondetU16Reg>
      %26 = zmir.literal 4 : !zmir.val
      %27 = zmir.read_field %arg0[@"$temp_8"] : !zmir.component<@AddrDecompose>, !zmir.val
      %28 = zmir.read_field %arg0[@"$temp_9"] : !zmir.component<@AddrDecompose>, !zmir.val
      %29 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %30 = builtin.unrealized_conversion_cast %29 : !zmir.pending to !zmir.val
      zmir.constrain %28 = %30 : !zmir.val, !zmir.val
      %31 = zmir.literal 16384 : !zmir.val
      %32 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %33 = zmir.read_field %arg0[@"$temp_10"] : !zmir.component<@AddrDecompose>, !zmir.val
      %34 = zmir.read_field %arg0[@addr] : !zmir.component<@AddrDecompose>, !zmir.val
      %35 = zmir.read_field %arg0[@"$super"] : !zmir.component<@AddrDecompose>, !zmir.val
      return
    }
  }
  zmir.split_component @AddrDecomposeBits attributes {name = "AddrDecomposeBits"} {
    zmir.field @"$super" : !zmir.val
    zmir.field @"$temp_13" : !zmir.val
    zmir.field @"$temp_12" : !zmir.val
    zmir.field @"$temp_11" : !zmir.val
    zmir.field @"$temp_10" : !zmir.component<@Div>
    zmir.field @"$temp_9" : !zmir.val
    zmir.field @"$temp_8" : !zmir.component<@IsZero>
    zmir.field @"$temp_7" : !zmir.val
    zmir.field @"$temp_6" : !zmir.val
    zmir.field @"$temp_5" : !zmir.val
    zmir.field @"$temp_4" : !zmir.val
    zmir.field @"$temp_3" : !zmir.val
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.component<@Div>
    zmir.field @"$temp_0" : !zmir.component<@BitAnd>
    zmir.field @"$temp" : !zmir.component<@BitAnd>
    zmir.field @addr : !zmir.val
    zmir.field @med14 : !zmir.component<@NondetU16Reg>
    zmir.field @upperDiff : !zmir.component<@U16Reg>
    zmir.field @low2 : !zmir.val
    zmir.field @low1 : !zmir.component<@NondetBitReg>
    zmir.field @low0 : !zmir.component<@NondetBitReg>
    func.func nested @compute(%arg0: !zmir.component<@ValU32>, %arg1: !zmir.val) -> !zmir.component<@AddrDecomposeBits> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.val to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %5 = zmir.literal 1 : !zmir.val
      %6 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      %7 = zmir.constructor @BitAnd : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %8 = call_indirect %7(%6, %5) {writes_into = "$temp"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %9 = zmir.self : !zmir.component<@AddrDecomposeBits>
      zmir.write_field %9[@"$temp"] = %8 : <@AddrDecomposeBits>, !zmir.component<@BitAnd>
      %10 = builtin.unrealized_conversion_cast %8 : !zmir.component<@BitAnd> to !zmir.pending
      %11 = builtin.unrealized_conversion_cast %10 : !zmir.pending to !zmir.val
      %12 = zmir.constructor @NondetBitReg : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %13 = call_indirect %12(%11) {writes_into = "low0"} : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %14 = builtin.unrealized_conversion_cast %13 : !zmir.component<@NondetBitReg> to !zmir.pending
      zmir.write_field %9[@low0] = %13 : <@AddrDecomposeBits>, !zmir.component<@NondetBitReg>
      %15 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %16 = zmir.literal 2 : !zmir.val
      %17 = builtin.unrealized_conversion_cast %15 : !zmir.pending to !zmir.val
      %18 = call_indirect %7(%17, %16) {writes_into = "$temp_0"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      zmir.write_field %9[@"$temp_0"] = %18 : <@AddrDecomposeBits>, !zmir.component<@BitAnd>
      %19 = builtin.unrealized_conversion_cast %18 : !zmir.component<@BitAnd> to !zmir.pending
      %20 = zmir.literal 2 : !zmir.val
      %21 = builtin.unrealized_conversion_cast %19 : !zmir.pending to !zmir.val
      %22 = zmir.constructor @Div : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      %23 = call_indirect %22(%21, %20) {writes_into = "$temp_1"} : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      zmir.write_field %9[@"$temp_1"] = %23 : <@AddrDecomposeBits>, !zmir.component<@Div>
      %24 = builtin.unrealized_conversion_cast %23 : !zmir.component<@Div> to !zmir.pending
      %25 = builtin.unrealized_conversion_cast %24 : !zmir.pending to !zmir.val
      %26 = call_indirect %12(%25) {writes_into = "low1"} : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %27 = builtin.unrealized_conversion_cast %26 : !zmir.component<@NondetBitReg> to !zmir.pending
      zmir.write_field %9[@low1] = %26 : <@AddrDecomposeBits>, !zmir.component<@NondetBitReg>
      %28 = zmir.literal 2 : !zmir.val
      %29 = builtin.unrealized_conversion_cast %27 : !zmir.pending to !zmir.val
      %30 = zmir.mul %29 : !zmir.val, %28 : !zmir.val
      zmir.write_field %9[@"$temp_2"] = %30 : <@AddrDecomposeBits>, !zmir.val
      %31 = builtin.unrealized_conversion_cast %14 : !zmir.pending to !zmir.val
      %32 = zmir.add %30 : !zmir.val, %31 : !zmir.val
      zmir.write_field %9[@low2] = %32 : <@AddrDecomposeBits>, !zmir.val
      %33 = zmir.literal 65535 : !zmir.val
      %34 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.val
      %35 = zmir.mul %34 : !zmir.val, %33 : !zmir.val
      zmir.write_field %9[@"$temp_3"] = %35 : <@AddrDecomposeBits>, !zmir.val
      %36 = zmir.literal 1 : !zmir.val
      %37 = zmir.sub %36 : !zmir.val, %34 : !zmir.val
      zmir.write_field %9[@"$temp_4"] = %37 : <@AddrDecomposeBits>, !zmir.val
      %38 = zmir.literal 49151 : !zmir.val
      %39 = zmir.mul %37 : !zmir.val, %38 : !zmir.val
      zmir.write_field %9[@"$temp_5"] = %39 : <@AddrDecomposeBits>, !zmir.val
      %40 = zmir.add %35 : !zmir.val, %39 : !zmir.val
      zmir.write_field %9[@"$temp_6"] = %40 : <@AddrDecomposeBits>, !zmir.val
      %41 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %42 = builtin.unrealized_conversion_cast %41 : !zmir.pending to !zmir.val
      %43 = zmir.sub %40 : !zmir.val, %42 : !zmir.val
      zmir.write_field %9[@"$temp_7"] = %43 : <@AddrDecomposeBits>, !zmir.val
      %44 = zmir.constructor @U16Reg : (!zmir.val) -> !zmir.component<@U16Reg>
      %45 = call_indirect %44(%43) {writes_into = "upperDiff"} : (!zmir.val) -> !zmir.component<@U16Reg>
      zmir.write_field %9[@upperDiff] = %45 : <@AddrDecomposeBits>, !zmir.component<@U16Reg>
      %46 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %47 = builtin.unrealized_conversion_cast %46 : !zmir.pending to !zmir.val
      %48 = zmir.constructor @IsZero : (!zmir.val) -> !zmir.component<@IsZero>
      %49 = call_indirect %48(%47) {writes_into = "$temp_8"} : (!zmir.val) -> !zmir.component<@IsZero>
      zmir.write_field %9[@"$temp_8"] = %49 : <@AddrDecomposeBits>, !zmir.component<@IsZero>
      %50 = zmir.literal 0 : !zmir.val
      %51 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %52 = builtin.unrealized_conversion_cast %51 : !zmir.pending to !zmir.val
      %53 = zmir.sub %52 : !zmir.val, %32 : !zmir.val
      zmir.write_field %9[@"$temp_9"] = %53 : <@AddrDecomposeBits>, !zmir.val
      %54 = zmir.literal 4 : !zmir.val
      %55 = call_indirect %22(%53, %54) {writes_into = "$temp_10"} : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      zmir.write_field %9[@"$temp_10"] = %55 : <@AddrDecomposeBits>, !zmir.component<@Div>
      %56 = builtin.unrealized_conversion_cast %55 : !zmir.component<@Div> to !zmir.pending
      %57 = builtin.unrealized_conversion_cast %56 : !zmir.pending to !zmir.val
      %58 = zmir.constructor @NondetU16Reg : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %59 = call_indirect %58(%57) {writes_into = "med14"} : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %60 = builtin.unrealized_conversion_cast %59 : !zmir.component<@NondetU16Reg> to !zmir.pending
      zmir.write_field %9[@med14] = %59 : <@AddrDecomposeBits>, !zmir.component<@NondetU16Reg>
      %61 = zmir.literal 4 : !zmir.val
      %62 = builtin.unrealized_conversion_cast %60 : !zmir.pending to !zmir.val
      %63 = zmir.mul %62 : !zmir.val, %61 : !zmir.val
      zmir.write_field %9[@"$temp_11"] = %63 : <@AddrDecomposeBits>, !zmir.val
      %64 = zmir.add %63 : !zmir.val, %32 : !zmir.val
      zmir.write_field %9[@"$temp_12"] = %64 : <@AddrDecomposeBits>, !zmir.val
      %65 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %66 = zmir.literal 16384 : !zmir.val
      %67 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %68 = builtin.unrealized_conversion_cast %67 : !zmir.pending to !zmir.val
      %69 = zmir.mul %66 : !zmir.val, %68 : !zmir.val
      zmir.write_field %9[@"$temp_13"] = %69 : <@AddrDecomposeBits>, !zmir.val
      %70 = zmir.add %69 : !zmir.val, %62 : !zmir.val
      zmir.write_field %9[@addr] = %70 : <@AddrDecomposeBits>, !zmir.val
      zmir.write_field %9[@"$super"] = %70 : <@AddrDecomposeBits>, !zmir.val
      return %9 : !zmir.component<@AddrDecomposeBits>
    }
    func.func nested @constrain(%arg0: !zmir.component<@AddrDecomposeBits>, %arg1: !zmir.component<@ValU32>, %arg2: !zmir.val) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %3 = zmir.literal 1 : !zmir.val
      %4 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@AddrDecomposeBits>, !zmir.component<@BitAnd>
      %5 = zmir.read_field %arg0[@low0] : !zmir.component<@AddrDecomposeBits>, !zmir.component<@NondetBitReg>
      %6 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %7 = zmir.literal 2 : !zmir.val
      %8 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@AddrDecomposeBits>, !zmir.component<@BitAnd>
      %9 = zmir.literal 2 : !zmir.val
      %10 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@AddrDecomposeBits>, !zmir.component<@Div>
      %11 = zmir.read_field %arg0[@low1] : !zmir.component<@AddrDecomposeBits>, !zmir.component<@NondetBitReg>
      %12 = zmir.literal 2 : !zmir.val
      %13 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %14 = zmir.read_field %arg0[@low2] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %15 = zmir.literal 65535 : !zmir.val
      %16 = zmir.read_field %arg0[@"$temp_3"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %17 = zmir.literal 1 : !zmir.val
      %18 = zmir.read_field %arg0[@"$temp_4"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %19 = zmir.literal 49151 : !zmir.val
      %20 = zmir.read_field %arg0[@"$temp_5"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %21 = zmir.read_field %arg0[@"$temp_6"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %22 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %23 = zmir.read_field %arg0[@"$temp_7"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %24 = zmir.read_field %arg0[@upperDiff] : !zmir.component<@AddrDecomposeBits>, !zmir.component<@U16Reg>
      %25 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %26 = zmir.read_field %arg0[@"$temp_8"] : !zmir.component<@AddrDecomposeBits>, !zmir.component<@IsZero>
      %27 = builtin.unrealized_conversion_cast %26 : !zmir.component<@IsZero> to !zmir.pending
      %28 = zmir.literal 0 : !zmir.val
      %29 = builtin.unrealized_conversion_cast %27 : !zmir.pending to !zmir.val
      zmir.constrain %29 = %28 : !zmir.val, !zmir.val
      %30 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %31 = zmir.read_field %arg0[@"$temp_9"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %32 = zmir.literal 4 : !zmir.val
      %33 = zmir.read_field %arg0[@"$temp_10"] : !zmir.component<@AddrDecomposeBits>, !zmir.component<@Div>
      %34 = zmir.read_field %arg0[@med14] : !zmir.component<@AddrDecomposeBits>, !zmir.component<@NondetU16Reg>
      %35 = zmir.literal 4 : !zmir.val
      %36 = zmir.read_field %arg0[@"$temp_11"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %37 = zmir.read_field %arg0[@"$temp_12"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %38 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %39 = builtin.unrealized_conversion_cast %38 : !zmir.pending to !zmir.val
      zmir.constrain %37 = %39 : !zmir.val, !zmir.val
      %40 = zmir.literal 16384 : !zmir.val
      %41 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %42 = zmir.read_field %arg0[@"$temp_13"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %43 = zmir.read_field %arg0[@addr] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      %44 = zmir.read_field %arg0[@"$super"] : !zmir.component<@AddrDecomposeBits>, !zmir.val
      return
    }
  }
  zmir.split_component @CondDenormed attributes {name = "CondDenormed"} {
    zmir.field @"$super" : !zmir.component<@DenormedValU32>
    zmir.field @"$temp_6" : !zmir.val
    zmir.field @"$temp_5" : !zmir.val
    zmir.field @"$temp_4" : !zmir.val
    zmir.field @"$temp_3" : !zmir.val
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.val
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.val
    func.func nested @compute(%arg0: !zmir.val, %arg1: !zmir.component<@DenormedValU32>, %arg2: !zmir.component<@DenormedValU32>) -> !zmir.component<@CondDenormed> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.val to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@DenormedValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = builtin.unrealized_conversion_cast %arg2 : !zmir.component<@DenormedValU32> to !zhl.expr
      %5 = builtin.unrealized_conversion_cast %4 : !zhl.expr to !zmir.pending
      %6 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %7 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.val
      %8 = builtin.unrealized_conversion_cast %6 : !zmir.pending to !zmir.val
      %9 = zmir.mul %7 : !zmir.val, %8 : !zmir.val
      %10 = zmir.self : !zmir.component<@CondDenormed>
      zmir.write_field %10[@"$temp"] = %9 : <@CondDenormed>, !zmir.val
      %11 = zmir.literal 1 : !zmir.val
      %12 = zmir.sub %11 : !zmir.val, %7 : !zmir.val
      zmir.write_field %10[@"$temp_0"] = %12 : <@CondDenormed>, !zmir.val
      %13 = zmir.read_field %5[@low] : !zmir.pending, !zmir.pending
      %14 = builtin.unrealized_conversion_cast %13 : !zmir.pending to !zmir.val
      %15 = zmir.mul %12 : !zmir.val, %14 : !zmir.val
      zmir.write_field %10[@"$temp_1"] = %15 : <@CondDenormed>, !zmir.val
      %16 = zmir.add %9 : !zmir.val, %15 : !zmir.val
      zmir.write_field %10[@"$temp_2"] = %16 : <@CondDenormed>, !zmir.val
      %17 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %18 = builtin.unrealized_conversion_cast %17 : !zmir.pending to !zmir.val
      %19 = zmir.mul %7 : !zmir.val, %18 : !zmir.val
      zmir.write_field %10[@"$temp_3"] = %19 : <@CondDenormed>, !zmir.val
      %20 = zmir.literal 1 : !zmir.val
      %21 = zmir.sub %20 : !zmir.val, %7 : !zmir.val
      zmir.write_field %10[@"$temp_4"] = %21 : <@CondDenormed>, !zmir.val
      %22 = zmir.read_field %5[@high] : !zmir.pending, !zmir.pending
      %23 = builtin.unrealized_conversion_cast %22 : !zmir.pending to !zmir.val
      %24 = zmir.mul %21 : !zmir.val, %23 : !zmir.val
      zmir.write_field %10[@"$temp_5"] = %24 : <@CondDenormed>, !zmir.val
      %25 = zmir.add %19 : !zmir.val, %24 : !zmir.val
      zmir.write_field %10[@"$temp_6"] = %25 : <@CondDenormed>, !zmir.val
      %26 = zmir.constructor @DenormedValU32 : (!zmir.val, !zmir.val) -> !zmir.component<@DenormedValU32>
      %27 = call_indirect %26(%16, %25) {writes_into = "$super"} : (!zmir.val, !zmir.val) -> !zmir.component<@DenormedValU32>
      zmir.write_field %10[@"$super"] = %27 : <@CondDenormed>, !zmir.component<@DenormedValU32>
      return %10 : !zmir.component<@CondDenormed>
    }
    func.func nested @constrain(%arg0: !zmir.component<@CondDenormed>, %arg1: !zmir.val, %arg2: !zmir.component<@DenormedValU32>, %arg3: !zmir.component<@DenormedValU32>) {
      %0 = builtin.unrealized_conversion_cast %arg2 : !zmir.component<@DenormedValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg3 : !zmir.component<@DenormedValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %5 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@CondDenormed>, !zmir.val
      %6 = zmir.literal 1 : !zmir.val
      %7 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@CondDenormed>, !zmir.val
      %8 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %9 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@CondDenormed>, !zmir.val
      %10 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@CondDenormed>, !zmir.val
      %11 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %12 = zmir.read_field %arg0[@"$temp_3"] : !zmir.component<@CondDenormed>, !zmir.val
      %13 = zmir.literal 1 : !zmir.val
      %14 = zmir.read_field %arg0[@"$temp_4"] : !zmir.component<@CondDenormed>, !zmir.val
      %15 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %16 = zmir.read_field %arg0[@"$temp_5"] : !zmir.component<@CondDenormed>, !zmir.val
      %17 = zmir.read_field %arg0[@"$temp_6"] : !zmir.component<@CondDenormed>, !zmir.val
      %18 = zmir.read_field %arg0[@"$super"] : !zmir.component<@CondDenormed>, !zmir.component<@DenormedValU32>
      return
    }
  }
  zmir.split_component @AssertEqU32 attributes {name = "AssertEqU32"} {
    zmir.field @"$super" : !zmir.component<@Component>
    func.func nested @compute(%arg0: !zmir.component<@ValU32>, %arg1: !zmir.component<@ValU32>) -> !zmir.component<@AssertEqU32> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %5 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %6 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %7 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %8 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %9 = call_indirect %8() {writes_into = "$super"} : () -> !zmir.component<@Component>
      %10 = zmir.self : !zmir.component<@AssertEqU32>
      zmir.write_field %10[@"$super"] = %9 : <@AssertEqU32>, !zmir.component<@Component>
      return %10 : !zmir.component<@AssertEqU32>
    }
    func.func nested @constrain(%arg0: !zmir.component<@AssertEqU32>, %arg1: !zmir.component<@ValU32>, %arg2: !zmir.component<@ValU32>) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg2 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %5 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %6 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      %7 = builtin.unrealized_conversion_cast %5 : !zmir.pending to !zmir.val
      zmir.constrain %6 = %7 : !zmir.val, !zmir.val
      %8 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %9 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %10 = builtin.unrealized_conversion_cast %8 : !zmir.pending to !zmir.val
      %11 = builtin.unrealized_conversion_cast %9 : !zmir.pending to !zmir.val
      zmir.constrain %10 = %11 : !zmir.val, !zmir.val
      %12 = zmir.read_field %arg0[@"$super"] : !zmir.component<@AssertEqU32>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @CmpEqual attributes {name = "CmpEqual"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @"$temp_1" : !zmir.val
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.val
    zmir.field @is_equal : !zmir.component<@Reg>
    zmir.field @high_same : !zmir.component<@IsZero>
    zmir.field @low_same : !zmir.component<@IsZero>
    func.func nested @compute(%arg0: !zmir.component<@ValU32>, %arg1: !zmir.component<@ValU32>) -> !zmir.component<@CmpEqual> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %5 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %6 = builtin.unrealized_conversion_cast %4 : !zmir.pending to !zmir.val
      %7 = builtin.unrealized_conversion_cast %5 : !zmir.pending to !zmir.val
      %8 = zmir.sub %6 : !zmir.val, %7 : !zmir.val
      %9 = zmir.self : !zmir.component<@CmpEqual>
      zmir.write_field %9[@"$temp"] = %8 : <@CmpEqual>, !zmir.val
      %10 = zmir.constructor @IsZero : (!zmir.val) -> !zmir.component<@IsZero>
      %11 = call_indirect %10(%8) {writes_into = "low_same"} : (!zmir.val) -> !zmir.component<@IsZero>
      %12 = builtin.unrealized_conversion_cast %11 : !zmir.component<@IsZero> to !zmir.pending
      zmir.write_field %9[@low_same] = %11 : <@CmpEqual>, !zmir.component<@IsZero>
      %13 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %14 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %15 = builtin.unrealized_conversion_cast %13 : !zmir.pending to !zmir.val
      %16 = builtin.unrealized_conversion_cast %14 : !zmir.pending to !zmir.val
      %17 = zmir.sub %15 : !zmir.val, %16 : !zmir.val
      zmir.write_field %9[@"$temp_0"] = %17 : <@CmpEqual>, !zmir.val
      %18 = call_indirect %10(%17) {writes_into = "high_same"} : (!zmir.val) -> !zmir.component<@IsZero>
      %19 = builtin.unrealized_conversion_cast %18 : !zmir.component<@IsZero> to !zmir.pending
      zmir.write_field %9[@high_same] = %18 : <@CmpEqual>, !zmir.component<@IsZero>
      %20 = builtin.unrealized_conversion_cast %12 : !zmir.pending to !zmir.val
      %21 = builtin.unrealized_conversion_cast %19 : !zmir.pending to !zmir.val
      %22 = zmir.mul %20 : !zmir.val, %21 : !zmir.val
      zmir.write_field %9[@"$temp_1"] = %22 : <@CmpEqual>, !zmir.val
      %23 = zmir.constructor @Reg : (!zmir.val) -> !zmir.component<@Reg>
      %24 = call_indirect %23(%22) {writes_into = "is_equal"} : (!zmir.val) -> !zmir.component<@Reg>
      zmir.write_field %9[@is_equal] = %24 : <@CmpEqual>, !zmir.component<@Reg>
      %25 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %26 = call_indirect %25() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %9[@"$super"] = %26 : <@CmpEqual>, !zmir.component<@Component>
      return %9 : !zmir.component<@CmpEqual>
    }
    func.func nested @constrain(%arg0: !zmir.component<@CmpEqual>, %arg1: !zmir.component<@ValU32>, %arg2: !zmir.component<@ValU32>) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg2 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = zmir.read_field %1[@low] : !zmir.pending, !zmir.pending
      %5 = zmir.read_field %3[@low] : !zmir.pending, !zmir.pending
      %6 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@CmpEqual>, !zmir.val
      %7 = zmir.read_field %arg0[@low_same] : !zmir.component<@CmpEqual>, !zmir.component<@IsZero>
      %8 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %9 = zmir.read_field %3[@high] : !zmir.pending, !zmir.pending
      %10 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@CmpEqual>, !zmir.val
      %11 = zmir.read_field %arg0[@high_same] : !zmir.component<@CmpEqual>, !zmir.component<@IsZero>
      %12 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@CmpEqual>, !zmir.val
      %13 = zmir.read_field %arg0[@is_equal] : !zmir.component<@CmpEqual>, !zmir.component<@Reg>
      %14 = zmir.read_field %arg0[@"$super"] : !zmir.component<@CmpEqual>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @CmpLessThanUnsigned attributes {name = "CmpLessThanUnsigned"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @"$temp" : !zmir.component<@SubU32>
    zmir.field @is_less_than : !zmir.val
    zmir.field @diff : !zmir.component<@NormalizeU32>
    func.func nested @compute(%arg0: !zmir.component<@ValU32>, %arg1: !zmir.component<@ValU32>) -> !zmir.component<@CmpLessThanUnsigned> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.component<@ValU32>
      %5 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.component<@ValU32>
      %6 = zmir.constructor @SubU32 : (!zmir.component<@ValU32>, !zmir.component<@ValU32>) -> !zmir.component<@SubU32>
      %7 = call_indirect %6(%4, %5) {writes_into = "$temp"} : (!zmir.component<@ValU32>, !zmir.component<@ValU32>) -> !zmir.component<@SubU32>
      %8 = zmir.self : !zmir.component<@CmpLessThanUnsigned>
      zmir.write_field %8[@"$temp"] = %7 : <@CmpLessThanUnsigned>, !zmir.component<@SubU32>
      %9 = builtin.unrealized_conversion_cast %7 : !zmir.component<@SubU32> to !zmir.pending
      %10 = builtin.unrealized_conversion_cast %9 : !zmir.pending to !zmir.component<@DenormedValU32>
      %11 = zmir.constructor @NormalizeU32 : (!zmir.component<@DenormedValU32>) -> !zmir.component<@NormalizeU32>
      %12 = call_indirect %11(%10) {writes_into = "diff"} : (!zmir.component<@DenormedValU32>) -> !zmir.component<@NormalizeU32>
      %13 = builtin.unrealized_conversion_cast %12 : !zmir.component<@NormalizeU32> to !zmir.pending
      zmir.write_field %8[@diff] = %12 : <@CmpLessThanUnsigned>, !zmir.component<@NormalizeU32>
      %14 = zmir.literal 1 : !zmir.val
      %15 = zmir.read_field %13[@carry] : !zmir.pending, !zmir.pending
      %16 = builtin.unrealized_conversion_cast %15 : !zmir.pending to !zmir.val
      %17 = zmir.sub %14 : !zmir.val, %16 : !zmir.val
      zmir.write_field %8[@is_less_than] = %17 : <@CmpLessThanUnsigned>, !zmir.val
      %18 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %19 = call_indirect %18() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %8[@"$super"] = %19 : <@CmpLessThanUnsigned>, !zmir.component<@Component>
      return %8 : !zmir.component<@CmpLessThanUnsigned>
    }
    func.func nested @constrain(%arg0: !zmir.component<@CmpLessThanUnsigned>, %arg1: !zmir.component<@ValU32>, %arg2: !zmir.component<@ValU32>) {
      %0 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@CmpLessThanUnsigned>, !zmir.component<@SubU32>
      %1 = zmir.read_field %arg0[@diff] : !zmir.component<@CmpLessThanUnsigned>, !zmir.component<@NormalizeU32>
      %2 = builtin.unrealized_conversion_cast %1 : !zmir.component<@NormalizeU32> to !zmir.pending
      %3 = zmir.literal 1 : !zmir.val
      %4 = zmir.read_field %2[@carry] : !zmir.pending, !zmir.pending
      %5 = zmir.read_field %arg0[@is_less_than] : !zmir.component<@CmpLessThanUnsigned>, !zmir.val
      %6 = zmir.read_field %arg0[@"$super"] : !zmir.component<@CmpLessThanUnsigned>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @GetSignU32 attributes {name = "GetSignU32"} {
    zmir.field @"$super" : !zmir.component<@NondetBitReg>
    zmir.field @"$temp_5" : !zmir.val
    zmir.field @"$temp_4" : !zmir.component<@Div>
    zmir.field @"$temp_3" : !zmir.val
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.component<@BitAnd>
    zmir.field @"$temp_0" : !zmir.component<@Div>
    zmir.field @"$temp" : !zmir.component<@BitAnd>
    zmir.field @rest_times_two : !zmir.component<@NondetU16Reg>
    zmir.field @sign : !zmir.component<@NondetBitReg>
    func.func nested @compute(%arg0: !zmir.component<@ValU32>) -> !zmir.component<@GetSignU32> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %3 = zmir.literal 32768 : !zmir.val
      %4 = builtin.unrealized_conversion_cast %2 : !zmir.pending to !zmir.val
      %5 = zmir.constructor @BitAnd : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %6 = call_indirect %5(%4, %3) {writes_into = "$temp"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      %7 = zmir.self : !zmir.component<@GetSignU32>
      zmir.write_field %7[@"$temp"] = %6 : <@GetSignU32>, !zmir.component<@BitAnd>
      %8 = builtin.unrealized_conversion_cast %6 : !zmir.component<@BitAnd> to !zmir.pending
      %9 = zmir.literal 32768 : !zmir.val
      %10 = builtin.unrealized_conversion_cast %8 : !zmir.pending to !zmir.val
      %11 = zmir.constructor @Div : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      %12 = call_indirect %11(%10, %9) {writes_into = "$temp_0"} : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      zmir.write_field %7[@"$temp_0"] = %12 : <@GetSignU32>, !zmir.component<@Div>
      %13 = builtin.unrealized_conversion_cast %12 : !zmir.component<@Div> to !zmir.pending
      %14 = builtin.unrealized_conversion_cast %13 : !zmir.pending to !zmir.val
      %15 = zmir.constructor @NondetBitReg : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %16 = call_indirect %15(%14) {writes_into = "$super"} : (!zmir.val) -> !zmir.component<@NondetBitReg>
      %17 = builtin.unrealized_conversion_cast %16 : !zmir.component<@NondetBitReg> to !zmir.pending
      zmir.write_field %7[@sign] = %16 : <@GetSignU32>, !zmir.component<@NondetBitReg>
      %18 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %19 = zmir.literal 32767 : !zmir.val
      %20 = builtin.unrealized_conversion_cast %18 : !zmir.pending to !zmir.val
      %21 = call_indirect %5(%20, %19) {writes_into = "$temp_1"} : (!zmir.val, !zmir.val) -> !zmir.component<@BitAnd>
      zmir.write_field %7[@"$temp_1"] = %21 : <@GetSignU32>, !zmir.component<@BitAnd>
      %22 = builtin.unrealized_conversion_cast %21 : !zmir.component<@BitAnd> to !zmir.pending
      %23 = zmir.literal 2 : !zmir.val
      %24 = builtin.unrealized_conversion_cast %22 : !zmir.pending to !zmir.val
      %25 = zmir.mul %24 : !zmir.val, %23 : !zmir.val
      zmir.write_field %7[@"$temp_2"] = %25 : <@GetSignU32>, !zmir.val
      %26 = zmir.constructor @NondetU16Reg : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %27 = call_indirect %26(%25) {writes_into = "rest_times_two"} : (!zmir.val) -> !zmir.component<@NondetU16Reg>
      %28 = builtin.unrealized_conversion_cast %27 : !zmir.component<@NondetU16Reg> to !zmir.pending
      zmir.write_field %7[@rest_times_two] = %27 : <@GetSignU32>, !zmir.component<@NondetU16Reg>
      %29 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %30 = zmir.literal 32768 : !zmir.val
      %31 = builtin.unrealized_conversion_cast %17 : !zmir.pending to !zmir.val
      %32 = zmir.mul %30 : !zmir.val, %31 : !zmir.val
      zmir.write_field %7[@"$temp_3"] = %32 : <@GetSignU32>, !zmir.val
      %33 = zmir.literal 2 : !zmir.val
      %34 = builtin.unrealized_conversion_cast %28 : !zmir.pending to !zmir.val
      %35 = call_indirect %11(%34, %33) {writes_into = "$temp_4"} : (!zmir.val, !zmir.val) -> !zmir.component<@Div>
      zmir.write_field %7[@"$temp_4"] = %35 : <@GetSignU32>, !zmir.component<@Div>
      %36 = builtin.unrealized_conversion_cast %35 : !zmir.component<@Div> to !zmir.pending
      %37 = builtin.unrealized_conversion_cast %36 : !zmir.pending to !zmir.val
      %38 = zmir.add %32 : !zmir.val, %37 : !zmir.val
      zmir.write_field %7[@"$temp_5"] = %38 : <@GetSignU32>, !zmir.val
      zmir.write_field %7[@"$super"] = %16 : <@GetSignU32>, !zmir.component<@NondetBitReg>
      return %7 : !zmir.component<@GetSignU32>
    }
    func.func nested @constrain(%arg0: !zmir.component<@GetSignU32>, %arg1: !zmir.component<@ValU32>) {
      %0 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %3 = zmir.literal 32768 : !zmir.val
      %4 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@GetSignU32>, !zmir.component<@BitAnd>
      %5 = zmir.literal 32768 : !zmir.val
      %6 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@GetSignU32>, !zmir.component<@Div>
      %7 = zmir.read_field %arg0[@sign] : !zmir.component<@GetSignU32>, !zmir.component<@NondetBitReg>
      %8 = zmir.read_field %arg0[@"$super"] : !zmir.component<@GetSignU32>, !zmir.component<@NondetBitReg>
      %9 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %10 = zmir.literal 32767 : !zmir.val
      %11 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@GetSignU32>, !zmir.component<@BitAnd>
      %12 = zmir.literal 2 : !zmir.val
      %13 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@GetSignU32>, !zmir.val
      %14 = zmir.read_field %arg0[@rest_times_two] : !zmir.component<@GetSignU32>, !zmir.component<@NondetU16Reg>
      %15 = zmir.read_field %1[@high] : !zmir.pending, !zmir.pending
      %16 = zmir.literal 32768 : !zmir.val
      %17 = zmir.read_field %arg0[@"$temp_3"] : !zmir.component<@GetSignU32>, !zmir.val
      %18 = zmir.literal 2 : !zmir.val
      %19 = zmir.read_field %arg0[@"$temp_4"] : !zmir.component<@GetSignU32>, !zmir.component<@Div>
      %20 = zmir.read_field %arg0[@"$temp_5"] : !zmir.component<@GetSignU32>, !zmir.val
      %21 = builtin.unrealized_conversion_cast %15 : !zmir.pending to !zmir.val
      zmir.constrain %21 = %20 : !zmir.val, !zmir.val
      return
    }
  }
  zmir.split_component @CmpLessThan attributes {name = "CmpLessThan"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @"$temp_11" : !zmir.val
    zmir.field @"$temp_10" : !zmir.val
    zmir.field @"$temp_9" : !zmir.val
    zmir.field @"$temp_8" : !zmir.val
    zmir.field @"$temp_7" : !zmir.val
    zmir.field @"$temp_6" : !zmir.val
    zmir.field @"$temp_5" : !zmir.val
    zmir.field @"$temp_4" : !zmir.val
    zmir.field @"$temp_3" : !zmir.val
    zmir.field @"$temp_2" : !zmir.val
    zmir.field @"$temp_1" : !zmir.val
    zmir.field @"$temp_0" : !zmir.val
    zmir.field @"$temp" : !zmir.component<@SubU32>
    zmir.field @is_less_than : !zmir.component<@Reg>
    zmir.field @overflow : !zmir.component<@Reg>
    zmir.field @s3 : !zmir.component<@GetSignU32>
    zmir.field @s2 : !zmir.component<@GetSignU32>
    zmir.field @s1 : !zmir.component<@GetSignU32>
    zmir.field @diff : !zmir.component<@NormalizeU32>
    func.func nested @compute(%arg0: !zmir.component<@ValU32>, %arg1: !zmir.component<@ValU32>) -> !zmir.component<@CmpLessThan> {
      %0 = builtin.unrealized_conversion_cast %arg0 : !zmir.component<@ValU32> to !zhl.expr
      %1 = builtin.unrealized_conversion_cast %0 : !zhl.expr to !zmir.pending
      %2 = builtin.unrealized_conversion_cast %arg1 : !zmir.component<@ValU32> to !zhl.expr
      %3 = builtin.unrealized_conversion_cast %2 : !zhl.expr to !zmir.pending
      %4 = builtin.unrealized_conversion_cast %1 : !zmir.pending to !zmir.component<@ValU32>
      %5 = builtin.unrealized_conversion_cast %3 : !zmir.pending to !zmir.component<@ValU32>
      %6 = zmir.constructor @SubU32 : (!zmir.component<@ValU32>, !zmir.component<@ValU32>) -> !zmir.component<@SubU32>
      %7 = call_indirect %6(%4, %5) {writes_into = "$temp"} : (!zmir.component<@ValU32>, !zmir.component<@ValU32>) -> !zmir.component<@SubU32>
      %8 = zmir.self : !zmir.component<@CmpLessThan>
      zmir.write_field %8[@"$temp"] = %7 : <@CmpLessThan>, !zmir.component<@SubU32>
      %9 = builtin.unrealized_conversion_cast %7 : !zmir.component<@SubU32> to !zmir.pending
      %10 = builtin.unrealized_conversion_cast %9 : !zmir.pending to !zmir.component<@DenormedValU32>
      %11 = zmir.constructor @NormalizeU32 : (!zmir.component<@DenormedValU32>) -> !zmir.component<@NormalizeU32>
      %12 = call_indirect %11(%10) {writes_into = "diff"} : (!zmir.component<@DenormedValU32>) -> !zmir.component<@NormalizeU32>
      %13 = builtin.unrealized_conversion_cast %12 : !zmir.component<@NormalizeU32> to !zmir.pending
      zmir.write_field %8[@diff] = %12 : <@CmpLessThan>, !zmir.component<@NormalizeU32>
      %14 = zmir.constructor @GetSignU32 : (!zmir.component<@ValU32>) -> !zmir.component<@GetSignU32>
      %15 = call_indirect %14(%4) {writes_into = "s1"} : (!zmir.component<@ValU32>) -> !zmir.component<@GetSignU32>
      %16 = builtin.unrealized_conversion_cast %15 : !zmir.component<@GetSignU32> to !zmir.pending
      zmir.write_field %8[@s1] = %15 : <@CmpLessThan>, !zmir.component<@GetSignU32>
      %17 = call_indirect %14(%5) {writes_into = "s2"} : (!zmir.component<@ValU32>) -> !zmir.component<@GetSignU32>
      %18 = builtin.unrealized_conversion_cast %17 : !zmir.component<@GetSignU32> to !zmir.pending
      zmir.write_field %8[@s2] = %17 : <@CmpLessThan>, !zmir.component<@GetSignU32>
      %19 = builtin.unrealized_conversion_cast %13 : !zmir.pending to !zmir.component<@ValU32>
      %20 = call_indirect %14(%19) {writes_into = "s3"} : (!zmir.component<@ValU32>) -> !zmir.component<@GetSignU32>
      %21 = builtin.unrealized_conversion_cast %20 : !zmir.component<@GetSignU32> to !zmir.pending
      zmir.write_field %8[@s3] = %20 : <@CmpLessThan>, !zmir.component<@GetSignU32>
      %22 = zmir.literal 1 : !zmir.val
      %23 = builtin.unrealized_conversion_cast %18 : !zmir.pending to !zmir.val
      %24 = zmir.sub %22 : !zmir.val, %23 : !zmir.val
      zmir.write_field %8[@"$temp_0"] = %24 : <@CmpLessThan>, !zmir.val
      %25 = builtin.unrealized_conversion_cast %16 : !zmir.pending to !zmir.val
      %26 = zmir.mul %25 : !zmir.val, %24 : !zmir.val
      zmir.write_field %8[@"$temp_1"] = %26 : <@CmpLessThan>, !zmir.val
      %27 = zmir.literal 1 : !zmir.val
      %28 = builtin.unrealized_conversion_cast %21 : !zmir.pending to !zmir.val
      %29 = zmir.sub %27 : !zmir.val, %28 : !zmir.val
      zmir.write_field %8[@"$temp_2"] = %29 : <@CmpLessThan>, !zmir.val
      %30 = zmir.mul %26 : !zmir.val, %29 : !zmir.val
      zmir.write_field %8[@"$temp_3"] = %30 : <@CmpLessThan>, !zmir.val
      %31 = zmir.literal 1 : !zmir.val
      %32 = zmir.sub %31 : !zmir.val, %25 : !zmir.val
      zmir.write_field %8[@"$temp_4"] = %32 : <@CmpLessThan>, !zmir.val
      %33 = zmir.mul %32 : !zmir.val, %23 : !zmir.val
      zmir.write_field %8[@"$temp_5"] = %33 : <@CmpLessThan>, !zmir.val
      %34 = zmir.mul %33 : !zmir.val, %28 : !zmir.val
      zmir.write_field %8[@"$temp_6"] = %34 : <@CmpLessThan>, !zmir.val
      %35 = zmir.add %30 : !zmir.val, %34 : !zmir.val
      zmir.write_field %8[@"$temp_7"] = %35 : <@CmpLessThan>, !zmir.val
      %36 = zmir.constructor @Reg : (!zmir.val) -> !zmir.component<@Reg>
      %37 = call_indirect %36(%35) {writes_into = "overflow"} : (!zmir.val) -> !zmir.component<@Reg>
      %38 = builtin.unrealized_conversion_cast %37 : !zmir.component<@Reg> to !zmir.pending
      zmir.write_field %8[@overflow] = %37 : <@CmpLessThan>, !zmir.component<@Reg>
      %39 = builtin.unrealized_conversion_cast %38 : !zmir.pending to !zmir.val
      %40 = zmir.add %39 : !zmir.val, %28 : !zmir.val
      zmir.write_field %8[@"$temp_8"] = %40 : <@CmpLessThan>, !zmir.val
      %41 = zmir.literal 2 : !zmir.val
      %42 = zmir.mul %41 : !zmir.val, %39 : !zmir.val
      zmir.write_field %8[@"$temp_9"] = %42 : <@CmpLessThan>, !zmir.val
      %43 = zmir.mul %42 : !zmir.val, %28 : !zmir.val
      zmir.write_field %8[@"$temp_10"] = %43 : <@CmpLessThan>, !zmir.val
      %44 = zmir.sub %40 : !zmir.val, %43 : !zmir.val
      zmir.write_field %8[@"$temp_11"] = %44 : <@CmpLessThan>, !zmir.val
      %45 = call_indirect %36(%44) {writes_into = "is_less_than"} : (!zmir.val) -> !zmir.component<@Reg>
      zmir.write_field %8[@is_less_than] = %45 : <@CmpLessThan>, !zmir.component<@Reg>
      %46 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %47 = call_indirect %46() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %8[@"$super"] = %47 : <@CmpLessThan>, !zmir.component<@Component>
      return %8 : !zmir.component<@CmpLessThan>
    }
    func.func nested @constrain(%arg0: !zmir.component<@CmpLessThan>, %arg1: !zmir.component<@ValU32>, %arg2: !zmir.component<@ValU32>) {
      %0 = zmir.read_field %arg0[@"$temp"] : !zmir.component<@CmpLessThan>, !zmir.component<@SubU32>
      %1 = zmir.read_field %arg0[@diff] : !zmir.component<@CmpLessThan>, !zmir.component<@NormalizeU32>
      %2 = zmir.read_field %arg0[@s1] : !zmir.component<@CmpLessThan>, !zmir.component<@GetSignU32>
      %3 = zmir.read_field %arg0[@s2] : !zmir.component<@CmpLessThan>, !zmir.component<@GetSignU32>
      %4 = zmir.read_field %arg0[@s3] : !zmir.component<@CmpLessThan>, !zmir.component<@GetSignU32>
      %5 = zmir.literal 1 : !zmir.val
      %6 = zmir.read_field %arg0[@"$temp_0"] : !zmir.component<@CmpLessThan>, !zmir.val
      %7 = zmir.read_field %arg0[@"$temp_1"] : !zmir.component<@CmpLessThan>, !zmir.val
      %8 = zmir.literal 1 : !zmir.val
      %9 = zmir.read_field %arg0[@"$temp_2"] : !zmir.component<@CmpLessThan>, !zmir.val
      %10 = zmir.read_field %arg0[@"$temp_3"] : !zmir.component<@CmpLessThan>, !zmir.val
      %11 = zmir.literal 1 : !zmir.val
      %12 = zmir.read_field %arg0[@"$temp_4"] : !zmir.component<@CmpLessThan>, !zmir.val
      %13 = zmir.read_field %arg0[@"$temp_5"] : !zmir.component<@CmpLessThan>, !zmir.val
      %14 = zmir.read_field %arg0[@"$temp_6"] : !zmir.component<@CmpLessThan>, !zmir.val
      %15 = zmir.read_field %arg0[@"$temp_7"] : !zmir.component<@CmpLessThan>, !zmir.val
      %16 = zmir.read_field %arg0[@overflow] : !zmir.component<@CmpLessThan>, !zmir.component<@Reg>
      %17 = zmir.read_field %arg0[@"$temp_8"] : !zmir.component<@CmpLessThan>, !zmir.val
      %18 = zmir.literal 2 : !zmir.val
      %19 = zmir.read_field %arg0[@"$temp_9"] : !zmir.component<@CmpLessThan>, !zmir.val
      %20 = zmir.read_field %arg0[@"$temp_10"] : !zmir.component<@CmpLessThan>, !zmir.val
      %21 = zmir.read_field %arg0[@"$temp_11"] : !zmir.component<@CmpLessThan>, !zmir.val
      %22 = zmir.read_field %arg0[@is_less_than] : !zmir.component<@CmpLessThan>, !zmir.component<@Reg>
      %23 = zmir.read_field %arg0[@"$super"] : !zmir.component<@CmpLessThan>, !zmir.component<@Component>
      return
    }
  }
  zmir.split_component @Top attributes {name = "Top"} {
    zmir.field @"$super" : !zmir.component<@Component>
    zmir.field @v : !zmir.component<@ValU32>
    func.func nested @compute() -> !zmir.component<@Top> {
      %0 = zmir.literal 10 : !zmir.val
      %1 = zmir.literal 20 : !zmir.val
      %2 = zmir.constructor @ValU32 : (!zmir.val, !zmir.val) -> !zmir.component<@ValU32>
      %3 = call_indirect %2(%0, %1) {writes_into = "v"} : (!zmir.val, !zmir.val) -> !zmir.component<@ValU32>
      %4 = zmir.self : !zmir.component<@Top>
      zmir.write_field %4[@v] = %3 : <@Top>, !zmir.component<@ValU32>
      %5 = zmir.constructor {builtin} @Component : () -> !zmir.component<@Component>
      %6 = call_indirect %5() {writes_into = "$super"} : () -> !zmir.component<@Component>
      zmir.write_field %4[@"$super"] = %6 : <@Top>, !zmir.component<@Component>
      return %4 : !zmir.component<@Top>
    }
    func.func nested @constrain(%arg0: !zmir.component<@Top>) {
      %0 = zmir.literal 10 : !zmir.val
      %1 = zmir.literal 20 : !zmir.val
      %2 = zmir.read_field %arg0[@v] : !zmir.component<@Top>, !zmir.component<@ValU32>
      %3 = zmir.read_field %arg0[@"$super"] : !zmir.component<@Top>, !zmir.component<@Component>
      return
    }
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(zmir-components-to-zkir)",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
