module attributes {veridise.lang = "llzk"} {
  llzk.struct @Component<[]> {
    func @compute() -> !llzk.struct<@Component<[]>> {
      %self = new_struct : <@Component<[]>>
      return %self : !llzk.struct<@Component<[]>>
    }
    func @constrain(%arg0: !llzk.struct<@Component<[]>>) {
      return
    }
  }
  llzk.struct @NondetReg<[]> {
    field @"$super" : !llzk.felt
    field @reg : !llzk.felt
    func @compute(%arg0: !llzk.felt) -> !llzk.struct<@NondetReg<[]>> {
      %self = new_struct : <@NondetReg<[]>>
      writef %self[@reg] = %arg0 : <@NondetReg<[]>>, !llzk.felt
      writef %self[@"$super"] = %arg0 : <@NondetReg<[]>>, !llzk.felt
      return %self : !llzk.struct<@NondetReg<[]>>
    }
    func @constrain(%arg0: !llzk.struct<@NondetReg<[]>>, %arg1: !llzk.felt) {
      return
    }
  }
  llzk.struct @Reg<[]> {
    field @"$super" : !llzk.struct<@NondetReg<[]>>
    field @reg : !llzk.struct<@NondetReg<[]>>
    func @compute(%arg0: !llzk.felt) -> !llzk.struct<@Reg<[]>> {
      %self = new_struct : <@Reg<[]>>
      %0 = call @NondetReg::@compute(%arg0) : (!llzk.felt) -> !llzk.struct<@NondetReg<[]>>
      writef %self[@reg] = %0 : <@Reg<[]>>, !llzk.struct<@NondetReg<[]>>
      %1 = readf %self[@reg] : <@Reg<[]>>, !llzk.struct<@NondetReg<[]>>
      %2 = readf %1[@"$super"] : <@NondetReg<[]>>, !llzk.felt
      writef %self[@"$super"] = %1 : <@Reg<[]>>, !llzk.struct<@NondetReg<[]>>
      return %self : !llzk.struct<@Reg<[]>>
    }
    func @constrain(%arg0: !llzk.struct<@Reg<[]>>, %arg1: !llzk.felt) {
      %0 = readf %arg0[@reg] : <@Reg<[]>>, !llzk.struct<@NondetReg<[]>>
      call @NondetReg::@constrain(%0, %arg1) : (!llzk.struct<@NondetReg<[]>>, !llzk.felt) -> ()
      %1 = readf %0[@"$super"] : <@NondetReg<[]>>, !llzk.felt
      emit_eq %arg1, %1 : !llzk.felt, !llzk.felt
      return
    }
  }
  llzk.struct @Div<[]> {
    field @"$super" : !llzk.felt
    field @reciprocal : !llzk.felt
    func @compute(%arg0: !llzk.felt, %arg1: !llzk.felt) -> !llzk.struct<@Div<[]>> {
      %self = new_struct : <@Div<[]>>
      %0 = inv %arg1 : !llzk.felt
      writef %self[@reciprocal] = %0 : <@Div<[]>>, !llzk.felt
      %1 = readf %self[@reciprocal] : <@Div<[]>>, !llzk.felt
      %2 = mul %1, %arg0 : !llzk.felt, !llzk.felt
      writef %self[@"$super"] = %2 : <@Div<[]>>, !llzk.felt
      return %self : !llzk.struct<@Div<[]>>
    }
    func @constrain(%arg0: !llzk.struct<@Div<[]>>, %arg1: !llzk.felt, %arg2: !llzk.felt) {
      %felt_const_1 = constfelt  1
      %0 = readf %arg0[@reciprocal] : <@Div<[]>>, !llzk.felt
      %1 = mul %0, %arg2 : !llzk.felt, !llzk.felt
      emit_eq %1, %felt_const_1 : !llzk.felt, !llzk.felt
      return
    }
  }
  llzk.struct @Log<[]> {
    field @"$super" : !llzk.struct<@Component<[]>>
    func @compute(%arg0: !llzk.string, %arg1: !llzk.array<-9223372036854775808 x !llzk.felt>) -> !llzk.struct<@Log<[]>> {
      %self = new_struct : <@Log<[]>>
      %0 = call @Log$$extern(%arg0, %arg1) : (!llzk.string, !llzk.array<-9223372036854775808 x !llzk.felt>) -> !llzk.struct<@Component<[]>>
      writef %self[@"$super"] = %0 : <@Log<[]>>, !llzk.struct<@Component<[]>>
      return %self : !llzk.struct<@Log<[]>>
    }
    func @constrain(%arg0: !llzk.struct<@Log<[]>>, %arg1: !llzk.string, %arg2: !llzk.array<-9223372036854775808 x !llzk.felt>) {
      %0 = call @Log$$extern(%arg1, %arg2) : (!llzk.string, !llzk.array<-9223372036854775808 x !llzk.felt>) -> !llzk.struct<@Component<[]>>
      return
    }
  }
  llzk.func private @Log$$extern(!llzk.string, !llzk.array<-9223372036854775808 x !llzk.felt>) -> !llzk.struct<@Component<[]>> attributes {extern}
}
