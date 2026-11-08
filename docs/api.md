# API

## Functions

```{eval-rst}
.. autofunction:: harm_analysis.harm_analysis

.. autofunction:: harm_analysis.spec_analysis
```

## Results

Both functions return a result object with one attribute per metric. Use
`to_dict()` to get the values as a dictionary, for example to print or serialize
them.

```{eval-rst}
.. autoclass:: harm_analysis.HarmResult
   :members: to_dict

.. autoclass:: harm_analysis.SpecResult
   :members: to_dict
```
