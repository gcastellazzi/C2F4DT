# Extensions

Drop plugin folders here. Each plugin should include a `plugin.py` exposing `PLUGIN` metadata:

```python
# extensions/my_plugin/plugin.py
from c2f4dt.plugins.manager import PluginMeta

PLUGIN = PluginMeta(
    name="My Plugin",
    order=50,
    requires=("Slicing",)  # optional dependencies by name
)
```
