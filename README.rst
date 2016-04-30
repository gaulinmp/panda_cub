Panda Cub
=================

Panda Cub is a package for random code snippets using pandas,
numpy, scipy, scikit-learn, etc.


Use
-----------------------------
Import modules with ``from panda_cub import stats``

Upon import (and where relevant), the module will monkey-patch pandas. 
So for example if you want to use ``t_test(df, 'year', 'decile', 'returns')``, after monkey-patching you can
call it directly on a dataframe: ``df.t_test('year', 'decile', 'returns')``
