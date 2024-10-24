module {
  zmir.component @Component {
    func.func @body () -> !zmir.component<@Component> {
      %0 = zmir.construct @Component () : () -> !zmir.component<@Component>
      func.return %0 : !zmir.component<@Component>
    }
  }
}
