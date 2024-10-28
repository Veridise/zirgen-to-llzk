module {
  zmir.component @InRange attributes {builtin} {
    func.func @body(%0: !zmir.val, %1: !zmir.val, %2: !zmir.val) -> !zmir.val {
      %3 = zmir.in_range %0 <= %1 < %2 : !zmir.val 
      func.return %3 : !zmir.val
    }
  }
}
