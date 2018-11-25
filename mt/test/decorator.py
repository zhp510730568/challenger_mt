def register_model(name=None):
  """Register a model. name defaults to class name snake-cased."""
  print("start decorator")
  def decorator(model_cls, registration_name=None):
    print("model_cls", model_cls)
    return model_cls

  # Handle if decorator was used without parens
  if callable(name):
    model_cls = name
    print('name: %s' % name)
    return decorator(model_cls, registration_name=None)

  return lambda model_cls: decorator(model_cls, name)