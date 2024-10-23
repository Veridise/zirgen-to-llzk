module {
  zmir.component @Component {
    zmir.body @Component$body () -> !zmir.component<@Component> {
      %0 = zmir.construct @Component () : () -> !zmir.component<@Component>
      zmir.return %0 : !zmir.component<@Component>
    }
  }
}
