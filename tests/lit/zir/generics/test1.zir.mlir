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
  llzk.struct @Head<[@T, @N]> {
    field @"$super" : !llzk.felt
    func @compute(%arg0: !llzk.array<@N x !llzk.tvar<@T>>) -> !llzk.struct<@Head<[@T, @N]>> {
      %felt_const_0 = constfelt  0
      %self = new_struct : <@Head<[@T, @N]>>
      %0 = read_const @N : !llzk.felt
      %1 = toindex %felt_const_0
      %2 = readarr %arg0[%1] : <@N x !llzk.tvar<@T>>, !llzk.tvar<@T>
      writef %self[@"$super"] = %2 : <@Head<[@T, @N]>>, !llzk.tvar<@T>
      return %self : !llzk.struct<@Head<[@T, @N]>>
    }
    func @constrain(%arg0: !llzk.struct<@Head<[@T, @N]>>, %arg1: !llzk.array<@N x !llzk.tvar<@T>>) {
      %felt_const_0 = constfelt  0
      %0 = read_const @N : !llzk.felt
      %1 = readf %arg0[@"$super"] : <@Head<[@T, @N]>>, !llzk.tvar<@T>
      %2 = toindex %felt_const_0
      %3 = readarr %arg1[%2] : <@N x !llzk.tvar<@T>>, !llzk.tvar<@T>
      return
    }
  }
  llzk.struct @A<[@N]> {
    field @"$super" : !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
    field @"$temp" : !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
    func @compute() -> !llzk.struct<@A<[@N]>> {
      %felt_const_2 = constfelt  2
      %felt_const_1 = constfelt  1
      %self = new_struct : <@A<[@N]>>
      %0 = read_const @N : !llzk.felt
      %1 = add %0, %felt_const_1 : !llzk.felt, !llzk.felt
      %2 = add %0, %felt_const_2 : !llzk.felt, !llzk.felt
      %array = new_array %0, %1, %2 : <3 x !llzk.felt>
      %3 = call @Head::@compute(%array) : (!llzk.array<3 x !llzk.felt>) -> !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      writef %self[@"$temp"] = %3 : <@A<[@N]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      %4 = readf %self[@"$temp"] : <@A<[@N]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      writef %self[@"$super"] = %4 : <@A<[@N]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      return %self : !llzk.struct<@A<[@N]>>
    }
    func @constrain(%arg0: !llzk.struct<@A<[@N]>>) {
      %felt_const_2 = constfelt  2
      %felt_const_1 = constfelt  1
      %0 = read_const @N : !llzk.felt
      %1 = add %0, %felt_const_1 : !llzk.felt, !llzk.felt
      %2 = add %0, %felt_const_2 : !llzk.felt, !llzk.felt
      %array = new_array %0, %1, %2 : <3 x !llzk.felt>
      %3 = readf %arg0[@"$temp"] : <@A<[@N]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      call @Head::@constrain(%3, %array) : (!llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>, !llzk.array<3 x !llzk.felt>) -> ()
      return
    }
  }
  llzk.struct @B<[]> {
    field @"$super" : !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
    field @"$temp_1" : !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
    field @"$temp_0" : !llzk.struct<@A<[!llzk.felt]>>
    field @"$temp" : !llzk.struct<@A<[!llzk.felt]>>
    func @compute() -> !llzk.struct<@B<[]>> {
      %self = new_struct : <@B<[]>>
      %0 = call @A::@compute() : () -> !llzk.struct<@A<[!llzk.felt]>>
      writef %self[@"$temp"] = %0 : <@B<[]>>, !llzk.struct<@A<[!llzk.felt]>>
      %1 = readf %self[@"$temp"] : <@B<[]>>, !llzk.struct<@A<[!llzk.felt]>>
      %2 = call @A::@compute() : () -> !llzk.struct<@A<[!llzk.felt]>>
      writef %self[@"$temp_0"] = %2 : <@B<[]>>, !llzk.struct<@A<[!llzk.felt]>>
      %3 = readf %self[@"$temp_0"] : <@B<[]>>, !llzk.struct<@A<[!llzk.felt]>>
      %array = new_array %1, %3 : <2 x !llzk.struct<@A<[!llzk.felt]>>>
      %4 = call @Head::@compute(%array) : (!llzk.array<2 x !llzk.struct<@A<[!llzk.felt]>>>) -> !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      writef %self[@"$temp_1"] = %4 : <@B<[]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      %5 = readf %self[@"$temp_1"] : <@B<[]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      writef %self[@"$super"] = %5 : <@B<[]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      return %self : !llzk.struct<@B<[]>>
    }
    func @constrain(%arg0: !llzk.struct<@B<[]>>) {
      %0 = readf %arg0[@"$temp"] : <@B<[]>>, !llzk.struct<@A<[!llzk.felt]>>
      call @A::@constrain(%0) : (!llzk.struct<@A<[!llzk.felt]>>) -> ()
      %1 = readf %arg0[@"$temp_0"] : <@B<[]>>, !llzk.struct<@A<[!llzk.felt]>>
      call @A::@constrain(%1) : (!llzk.struct<@A<[!llzk.felt]>>) -> ()
      %array = new_array %0, %1 : <2 x !llzk.struct<@A<[!llzk.felt]>>>
      %2 = readf %arg0[@"$temp_1"] : <@B<[]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      call @Head::@constrain(%2, %array) : (!llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>, !llzk.array<2 x !llzk.struct<@A<[!llzk.felt]>>>) -> ()
      return
    }
  }
  llzk.struct @Top<[]> {
    field @"$super" : !llzk.struct<@Component<[]>>
    field @"$temp" : !llzk.struct<@Component<[]>>
    field @b : !llzk.struct<@B<[]>>
    field @a : !llzk.struct<@A<[!llzk.felt]>>
    func @compute() -> !llzk.struct<@Top<[]>> {
      %self = new_struct : <@Top<[]>>
      %0 = call @A::@compute() : () -> !llzk.struct<@A<[!llzk.felt]>>
      writef %self[@a] = %0 : <@Top<[]>>, !llzk.struct<@A<[!llzk.felt]>>
      %1 = readf %self[@a] : <@Top<[]>>, !llzk.struct<@A<[!llzk.felt]>>
      %2 = call @B::@compute() : () -> !llzk.struct<@B<[]>>
      writef %self[@b] = %2 : <@Top<[]>>, !llzk.struct<@B<[]>>
      %3 = readf %self[@b] : <@Top<[]>>, !llzk.struct<@B<[]>>
      %4 = readf %1[@"$super"] : <@A<[!llzk.felt]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      %5 = readf %4[@"$super"] : <@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>, !llzk.felt
      %6 = readf %3[@"$super"] : <@B<[]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      %7 = readf %6[@"$super"] : <@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>, !llzk.felt
      %8 = call @Component::@compute() : () -> !llzk.struct<@Component<[]>>
      writef %self[@"$temp"] = %8 : <@Top<[]>>, !llzk.struct<@Component<[]>>
      %9 = readf %self[@"$temp"] : <@Top<[]>>, !llzk.struct<@Component<[]>>
      writef %self[@"$super"] = %9 : <@Top<[]>>, !llzk.struct<@Component<[]>>
      return %self : !llzk.struct<@Top<[]>>
    }
    func @constrain(%arg0: !llzk.struct<@Top<[]>>) {
      %0 = readf %arg0[@a] : <@Top<[]>>, !llzk.struct<@A<[!llzk.felt]>>
      call @A::@constrain(%0) : (!llzk.struct<@A<[!llzk.felt]>>) -> ()
      %1 = readf %arg0[@b] : <@Top<[]>>, !llzk.struct<@B<[]>>
      call @B::@constrain(%1) : (!llzk.struct<@B<[]>>) -> ()
      %2 = readf %0[@"$super"] : <@A<[!llzk.felt]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      %3 = readf %2[@"$super"] : <@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>, !llzk.felt
      %4 = readf %1[@"$super"] : <@B<[]>>, !llzk.struct<@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>
      %5 = readf %4[@"$super"] : <@Head<[!llzk.struct<@Component<[]>>, !llzk.felt]>>, !llzk.felt
      emit_eq %3, %5 : !llzk.felt, !llzk.felt
      %6 = readf %arg0[@"$temp"] : <@Top<[]>>, !llzk.struct<@Component<[]>>
      call @Component::@constrain(%6) : (!llzk.struct<@Component<[]>>) -> ()
      return
    }
  }
}
