# builders.domain.renderability

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Renderable

A domain must inherit this class if it can be rendered with any kind of visualization.

### render <Badge text="Renderable" type="tip"/>

<skdecide-signature name= "render" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}, {'name': 'kwargs', 'annotation': 'Any'}], 'return': 'Any'}"></skdecide-signature>

Compute a visual render of the given memory (state or history), or the internal one if omitted.

By default, `Renderable.render()` provides some boilerplate code and internally calls `Renderable._render()`. The
boilerplate code automatically passes the `_memory` attribute instead of the memory parameter whenever the latter
is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
A render (e.g. image) or nothing (if the function handles the display directly).

### \_render <Badge text="Renderable" type="tip"/>

<skdecide-signature name= "_render" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}, {'name': 'kwargs', 'annotation': 'Any'}], 'return': 'Any'}"></skdecide-signature>

Compute a visual render of the given memory (state or history), or the internal one if omitted.

By default, `Renderable._render()` provides some boilerplate code and internally
calls `Renderable._render_from()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
A render (e.g. image) or nothing (if the function handles the display directly).

### \_render\_from <Badge text="Renderable" type="tip"/>

<skdecide-signature name= "_render_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'kwargs', 'annotation': 'Any'}], 'return': 'Any'}"></skdecide-signature>

Compute a visual render of the given memory (state or history).

This is a helper function called by default from `Renderable._render()`, the difference being that the
memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
A render (e.g. image) or nothing (if the function handles the display directly).

